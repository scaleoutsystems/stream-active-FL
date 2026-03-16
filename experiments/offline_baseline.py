"""
Offline baseline: multi-epoch shuffled training on the full dataset.

Establishes the performance ceiling for comparison with streaming experiments.
Trains a detection model on all training frames for multiple epochs using a
standard DataLoader with shuffle, then evaluates on the validation set.

Usage:
    python experiments/offline_baseline.py --config configs/offline_baseline.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import torch

warnings.filterwarnings("ignore", message="Can't initialize NVML")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from stream_active_fl.core import (
    CATEGORY_ID_TO_NAME,
    DetectionDataset,
    DetectionStream,
    build_class_mapping,
    detection_collate,
    get_detection_augmentation,
    get_detection_transforms,
)
from stream_active_fl.evaluation import evaluate_detection
from stream_active_fl.experiment import (
    build_detector_from_config,
    load_dataclass_config,
    resolve_manifest_path,
    setup_run_dir,
)
from stream_active_fl.logging import save_run_info
from stream_active_fl.training import bootstrap_train
from stream_active_fl.utils import set_seed


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OfflineBaselineConfig:
    """Configuration for the offline baseline experiment."""

    # Paths
    manifest_path: str = ""
    output_dir: str = "outputs/offline/baseline"

    # Model / classes
    num_classes: int = 11
    target_classes: Optional[List[str]] = None
    trainable_backbone_layers: int = 3
    image_min_size: int = 480
    image_max_size: int = 1600
    pretrained_backbone: bool = True
    pretrained_detector: bool = True
    load_checkpoint: Optional[str] = None

    # Training
    epochs: int = 30
    batch_size: int = 8
    lr: float = 4e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_workers: int = 2
    lr_scheduler: Literal["none", "cosine"] = "cosine"
    lr_warmup_epochs: int = 1

    # Augmentation
    augment: bool = True
    hflip_prob: float = 0.5
    color_jitter: bool = True

    # Evaluation
    eval_every_n_epochs: int = 1
    score_threshold: float = 0.3
    min_box_area: float = 64.0
    bootstrap_smoke_check_frames: int = 200
    bootstrap_smoke_score_threshold: float = 0.3
    bootstrap_smoke_min_map50: float = 0.005
    bootstrap_fail_on_smoke_check: bool = True

    # Reproducibility
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OfflineBaselineConfig":
        return load_dataclass_config(cls, path)


# =============================================================================
# Epoch-level CSV logger
# =============================================================================


class EpochLogger:
    """Simple CSV logger for per-epoch training loss and evaluation metrics."""

    def __init__(self, log_dir: Path, class_names: Optional[List[str]] = None):
        self.log_dir = log_dir
        self.csv_path = log_dir / "epochs.csv"
        names = class_names if class_names is not None else list(CATEGORY_ID_TO_NAME.values())
        per_class_cols = [f"AP_{name}" for name in names]
        self.fieldnames = [
            "epoch",
            "train_loss",
            "lr",
            "mAP",
            "mAP_50",
            "mAP_75",
            "num_items",
            "total_predictions",
            "total_ground_truth",
            *per_class_cols,
        ]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, epoch: int, train_loss: float, eval_metrics: Optional[dict] = None, lr: Optional[float] = None) -> None:
        row = {k: "" for k in self.fieldnames}
        row["epoch"] = str(epoch)
        row["train_loss"] = f"{train_loss:.6f}"
        if lr is not None:
            row["lr"] = f"{lr:.2e}"
        if eval_metrics:
            for k, v in eval_metrics.items():
                if k in row:
                    row[k] = f"{v:.4f}" if isinstance(v, float) else str(v)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


# =============================================================================
# Main
# =============================================================================


def main(config: OfflineBaselineConfig, config_path: Path, command: str) -> None:
    start_time = datetime.now()

    print("=" * 60)
    print("Offline Baseline")
    print("=" * 60)

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest_path = resolve_manifest_path(PROJECT_ROOT, config.manifest_path)
    run_dir = setup_run_dir(PROJECT_ROOT, config.output_dir, config_path)
    print(f"Run directory: {run_dir}")

    class_mapping = build_class_mapping(config.target_classes)
    if config.target_classes is not None:
        config.num_classes = class_mapping.num_classes
        print(f"Target classes ({len(class_mapping.names)}): {', '.join(class_mapping.names)}")

    # Transforms + augmentation
    train_transform, val_transform = get_detection_transforms()
    train_augmentation = None
    if config.augment:
        train_augmentation = get_detection_augmentation(
            hflip_prob=config.hflip_prob,
            color_jitter=config.color_jitter,
        )

    # Datasets
    train_dataset = DetectionDataset(
        manifest_path=manifest_path,
        split="train",
        transform=train_transform,
        augmentation=train_augmentation,
        min_box_area=config.min_box_area,
        target_classes=config.target_classes,
    )

    val_stream = DetectionStream(
        manifest_path=manifest_path,
        split="val",
        transform=val_transform,
        min_box_area=config.min_box_area,
        target_classes=config.target_classes,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=detection_collate,
    )

    # Model
    model = build_detector_from_config(config)

    if config.load_checkpoint:
        checkpoint_path = PROJECT_ROOT / config.load_checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # LR scheduler
    scheduler = None
    if config.lr_scheduler == "cosine":
        main_epochs = max(config.epochs - config.lr_warmup_epochs, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=config.lr * 0.01,
        )
        print(f"LR scheduler: cosine (warmup={config.lr_warmup_epochs}, "
              f"T_max={main_epochs}, eta_min={config.lr * 0.01:.1e})")

    epoch_logger = EpochLogger(run_dir, class_names=list(class_mapping.names))
    best_mAP = 0.0
    best_epoch = 0

    # =========================================================================
    # Training loop
    # =========================================================================

    print(f"\nTraining: {len(train_dataset)} frames, {config.epochs} epochs")
    print(f"Val set:  {len(val_stream)} frames")

    for epoch in range(1, config.epochs + 1):
        # Linear LR warmup
        if epoch <= config.lr_warmup_epochs:
            warmup_factor = epoch / max(config.lr_warmup_epochs, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * warmup_factor

        epoch_logs, _ = bootstrap_train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epochs=1,
            max_grad_norm=config.max_grad_norm,
            progress_bar=True,
        )
        epoch_loss = epoch_logs[0]["avg_loss"] if epoch_logs else float("nan")

        # Step scheduler after warmup
        if scheduler is not None and epoch > config.lr_warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  LR: {current_lr:.2e}")

        # Evaluate
        eval_metrics = None
        if epoch % config.eval_every_n_epochs == 0 or epoch == config.epochs:
            print(f"\nEvaluating after epoch {epoch}...")
            eval_metrics = evaluate_detection(
                model, val_stream, device, score_threshold=config.score_threshold,
            )
            mAP = eval_metrics["mAP"]
            print(f"  mAP: {mAP:.4f}  mAP@50: {eval_metrics.get('mAP_50', 0.0):.4f}  mAP@75: {eval_metrics.get('mAP_75', 0.0):.4f}")

            if epoch == 1:
                smoke_stream = DetectionStream(
                    manifest_path=manifest_path,
                    split="val",
                    transform=val_transform,
                    min_box_area=config.min_box_area,
                    frame_range=(0, config.bootstrap_smoke_check_frames),
                    target_classes=config.target_classes,
                    verbose=False,
                )
                smoke_metrics = evaluate_detection(
                    model,
                    smoke_stream,
                    device,
                    score_threshold=config.bootstrap_smoke_score_threshold,
                )
                print(
                    "Smoke-check:"
                    f" mAP={smoke_metrics.get('mAP', 0.0):.4f},"
                    f" mAP@50={smoke_metrics.get('mAP_50', 0.0):.4f},"
                    f" preds={int(smoke_metrics.get('total_predictions', 0.0))},"
                    f" gt={int(smoke_metrics.get('total_ground_truth', 0.0))},"
                    f" items={int(smoke_metrics.get('num_items', 0.0))}"
                )
                if (
                    config.bootstrap_fail_on_smoke_check
                    and smoke_metrics.get("mAP_50", 0.0) < config.bootstrap_smoke_min_map50
                ):
                    raise RuntimeError(
                        "Smoke-check failed after epoch 1: "
                        f"mAP@50={smoke_metrics.get('mAP_50', 0.0):.4f} < "
                        f"{config.bootstrap_smoke_min_map50:.4f}. "
                        "Adjust initialization or training strength."
                    )

            if mAP > best_mAP:
                best_mAP = mAP
                best_epoch = epoch
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": epoch, "mAP": mAP},
                    run_dir / "best_model.pt",
                )

        epoch_logger.log(epoch, epoch_loss, eval_metrics, lr=current_lr)

    # =========================================================================
    # Final summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("Offline Baseline Complete")
    print(f"  Best mAP: {best_mAP:.4f} (epoch {best_epoch})")
    print("=" * 60)

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "best_mAP": best_mAP,
            "best_epoch": best_epoch,
        },
        run_dir / "final_model.pt",
    )

    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        end_time=datetime.now(),
        best_epoch=best_epoch,
        best_metric=best_mAP,
        best_metric_name="val_mAP",
        dataset_info={
            "train_frames": len(train_dataset),
            "val_frames": len(val_stream),
        },
        repo_path=PROJECT_ROOT,
    )

    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline baseline experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    command = " ".join(sys.argv)
    config = OfflineBaselineConfig.from_yaml(config_path)
    main(config, config_path, command)
