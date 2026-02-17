"""
Offline detection baseline experiment.

Trains an FCOS detector on the full ZOD dataset (shuffled, multi-epoch)
to establish an upper-bound baseline for comparison with streaming methods.

Usage:
    python experiments/offline_detection.py --config configs/offline_detection.yaml
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

warnings.filterwarnings("ignore", message="Can't initialize NVML")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from stream_active_fl.core import (
    StreamingDataset,
    ZODDetectionDataset,
    detection_collate,
    get_detection_augmentation,
    get_detection_transforms,
)
from stream_active_fl.evaluation import evaluate_detection
from stream_active_fl.logging import create_run_dir, save_run_info
from stream_active_fl.models import Detector
from stream_active_fl.utils import set_seed, worker_init_fn


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OfflineDetectionConfig:
    """Offline detection experiment configuration."""

    # Paths
    dataset_root: str = "/mnt/pr_2018_scaleout_workdir/ZOD256/ZOD_640x360"
    annotations_dir: str = "data/annotations_640x360"
    output_dir: str = "outputs/offline_detection"

    # Dataset
    min_score: float = 0.5
    subsample_steps: int = 5

    # Model
    num_classes: int = 4
    trainable_backbone_layers: int = 0
    image_min_size: int = 360
    image_max_size: int = 640
    pretrained_backbone: bool = True

    # Training
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Augmentation
    augment: bool = True
    hflip_prob: float = 0.5
    color_jitter: bool = True

    # Evaluation
    eval_every_n_epochs: int = 1
    score_threshold: float = 0.3

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OfflineDetectionConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Training loop
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch, returning average loss."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        if batch is None:
            continue

        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": total_loss / max(num_batches, 1)}


# =============================================================================
# Main
# =============================================================================

def main(config: OfflineDetectionConfig, config_path: Path, command: str) -> None:
    """Run offline detection baseline training."""

    start_time = datetime.now()

    print("=" * 60)
    print("Offline Detection Baseline Training")
    print("=" * 60)

    # Setup
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve paths and create run directory
    annotations_dir = PROJECT_ROOT / config.annotations_dir
    base_output_dir = PROJECT_ROOT / config.output_dir
    run_dir = create_run_dir(base_output_dir)
    print(f"Run directory: {run_dir}")

    shutil.copy(config_path, run_dir / "config.yaml")

    # Transforms and augmentation
    train_transform, val_transform = get_detection_transforms()
    train_augmentation = None
    if config.augment:
        train_augmentation = get_detection_augmentation(
            hflip_prob=config.hflip_prob,
            color_jitter=config.color_jitter,
        )

    # Datasets
    print("\nLoading datasets...")
    train_dataset = ZODDetectionDataset(
        dataset_root=config.dataset_root,
        annotations_dir=annotations_dir,
        split="train",
        transform=train_transform,
        min_score=config.min_score,
        subsample_steps=config.subsample_steps,
        augmentation=train_augmentation,
        verbose=True,
    )

    # Validation uses StreamingDataset for evaluate_detection() compatibility
    val_stream = StreamingDataset(
        dataset_root=config.dataset_root,
        annotations_dir=annotations_dir,
        split="val",
        transform=val_transform,
        min_score=config.min_score,
        subsample_steps=config.subsample_steps,
        task="detection",
        verbose=True,
    )

    dataset_info = {
        "train_total": len(train_dataset),
        "val_total": len(val_stream),
        "task": "detection",
    }

    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        dataset_info=dataset_info,
        repo_path=PROJECT_ROOT,
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # Model
    print("\nInitializing detection model...")
    model = Detector(
        num_classes=config.num_classes,
        trainable_backbone_layers=config.trainable_backbone_layers,
        image_min_size=config.image_min_size,
        image_max_size=config.image_max_size,
        pretrained_backbone=config.pretrained_backbone,
    )
    model = model.to(device)
    print(model)

    # Optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Initialize CSV log
    checkpoints_file = run_dir / "checkpoints.csv"
    with open(checkpoints_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss",
            "eval_mAP", "eval_mAP_50", "eval_mAP_75",
            "eval_AP_person", "eval_AP_car", "eval_AP_traffic_light",
            "epoch_time_s",
        ])

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_mAP = 0.0
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            max_grad_norm=config.max_grad_norm,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{config.epochs} "
            f"[{epoch_time:.1f}s] "
            f"Train Loss: {train_metrics['loss']:.4f}"
        )

        # Evaluate
        eval_metrics: Optional[Dict[str, float]] = None
        if epoch % config.eval_every_n_epochs == 0:
            eval_metrics = evaluate_detection(
                model, val_stream, device,
                score_threshold=config.score_threshold,
            )

            print(
                f"         mAP: {eval_metrics['mAP']:.4f} "
                f"mAP@50: {eval_metrics['mAP_50']:.4f} "
                f"mAP@75: {eval_metrics['mAP_75']:.4f} "
                f"AP_person: {eval_metrics.get('AP_person', 0.0):.4f} "
                f"AP_car: {eval_metrics.get('AP_car', 0.0):.4f} "
                f"AP_tl: {eval_metrics.get('AP_traffic_light', 0.0):.4f}"
            )

            # Save best model
            if eval_metrics["mAP"] > best_mAP:
                best_mAP = eval_metrics["mAP"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "eval_metrics": eval_metrics,
                        "config": config.__dict__,
                    },
                    run_dir / "best_model.pt",
                )
                print(f"         Saved best model (mAP={best_mAP:.4f})")

        # Log to CSV
        with open(checkpoints_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_metrics['loss']:.4f}",
                f"{eval_metrics['mAP']:.4f}" if eval_metrics else "",
                f"{eval_metrics['mAP_50']:.4f}" if eval_metrics else "",
                f"{eval_metrics['mAP_75']:.4f}" if eval_metrics else "",
                f"{eval_metrics.get('AP_person', 0.0):.4f}" if eval_metrics else "",
                f"{eval_metrics.get('AP_car', 0.0):.4f}" if eval_metrics else "",
                f"{eval_metrics.get('AP_traffic_light', 0.0):.4f}" if eval_metrics else "",
                f"{epoch_time:.1f}",
            ])

    # Save final model
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
        },
        run_dir / "final_model.pt",
    )

    # Save final run info
    end_time = datetime.now()
    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        end_time=end_time,
        best_epoch=best_epoch,
        best_metric=best_mAP,
        best_metric_name="val_mAP",
        dataset_info=dataset_info,
        repo_path=PROJECT_ROOT,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation mAP: {best_mAP:.4f} (epoch {best_epoch})")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline detection baseline training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/offline_detection.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    command = " ".join(sys.argv)

    config = OfflineDetectionConfig.from_yaml(config_path)
    main(config, config_path, command)
