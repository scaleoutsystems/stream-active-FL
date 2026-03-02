"""
Streaming detection experiment.

Two-phase pipeline:
  Phase 1 (Bootstrap): Multi-epoch training on the first N frames.
      Collects backbone embedding statistics for distribution-based filtering.
  Phase 2 (Streaming): Single-pass buffer-based training over the remaining
      frames with active filtering.

Bootstrap reuse: set bootstrap_run_dir in the config (or --bootstrap-run-dir
on the CLI) to skip Phase 1 and load the model + embeddings from a previous
run.  This saves hours when comparing filter policies.

Usage:
    python experiments/streaming_detection.py --config configs/detection/no_filter.yaml
    python experiments/streaming_detection.py --config configs/detection/distribution_filter.yaml \
        --bootstrap-run-dir outputs/no_filter/2026-03-02_09-32-13
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import yaml

warnings.filterwarnings("ignore", message="Can't initialize NVML")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from stream_active_fl.core import (
    DetectionDataset,
    DetectionStream,
    detection_collate,
    get_detection_augmentation,
    get_detection_transforms,
)
from stream_active_fl.evaluation import NoveltyTracker, evaluate_detection
from stream_active_fl.logging import StreamingMetricsLogger, create_run_dir, save_run_info
from stream_active_fl.memory import TrainingBuffer
from stream_active_fl.models import Detector
from stream_active_fl.policies import create_filter_policy
from stream_active_fl.training import bootstrap_train, collect_embeddings, train_on_stream
from stream_active_fl.utils import set_seed


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StreamingDetectionConfig:
    """Configuration for the streaming detection experiment."""

    # Paths
    manifest_path: str = ""
    output_dir: str = "outputs/streaming_detection"

    # Model
    num_classes: int = 11
    trainable_backbone_layers: int = 0
    image_min_size: int = 720
    image_max_size: int = 1280
    pretrained_backbone: bool = True
    pretrained_detector: bool = True
    load_checkpoint: Optional[str] = None

    # Bootstrap phase (skipped when bootstrap_run_dir is set)
    bootstrap_run_dir: Optional[str] = None
    bootstrap_frames: int = 5000
    bootstrap_epochs: int = 20
    bootstrap_batch_size: int = 8
    bootstrap_lr: float = 4e-4

    # Streaming phase — training
    streaming_lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    buffer_capacity: int = 16
    train_steps_per_buffer: int = 1

    # Augmentation
    augment: bool = True
    hflip_prob: float = 0.5
    color_jitter: bool = True

    # Filtering policy
    filter_policy: Literal["none", "distribution", "uncertainty", "gradient_norm"] = "none"
    accept_fraction: float = 0.3
    warmup_items: int = 200
    score_window_size: int = 500

    # Distribution-based policy
    distribution_mode: Literal["mahalanobis", "cosine", "knn"] = "mahalanobis"
    embedding_buffer_size: int = 1000
    knn_k: int = 10
    update_distribution_stats: bool = True

    # Uncertainty-based policy
    confidence_threshold: float = 0.1
    top_k_detections: int = 5

    # Gradient-norm policy
    norm_window_size: int = 500

    # Evaluation
    eval_every_n_items: int = 5000
    checkpoint_interval: int = 1000
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
    def from_yaml(cls, path: str | Path) -> "StreamingDetectionConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Main
# =============================================================================


def main(config: StreamingDetectionConfig, config_path: Path, command: str) -> None:
    start_time = datetime.now()

    print("=" * 60)
    print("Streaming Detection")
    print("=" * 60)

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest_path = Path(config.manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    base_output_dir = PROJECT_ROOT / config.output_dir
    run_dir = create_run_dir(base_output_dir)
    print(f"Run directory: {run_dir}")

    shutil.copy(config_path, run_dir / "config.yaml")

    # Transforms + augmentation
    train_transform, val_transform = get_detection_transforms()
    train_augmentation = None
    if config.augment:
        train_augmentation = get_detection_augmentation(
            hflip_prob=config.hflip_prob,
            color_jitter=config.color_jitter,
        )

    # =========================================================================
    # Phase 1: Bootstrap (run or reuse)
    # =========================================================================

    # Resolve bootstrap_run_dir (config or CLI override)
    bootstrap_source: Optional[Path] = None
    if config.bootstrap_run_dir:
        p = Path(config.bootstrap_run_dir)
        bootstrap_source = p if p.is_absolute() else PROJECT_ROOT / p

    if bootstrap_source is not None:
        # ----- Reuse bootstrap from a previous run -----
        model_path = bootstrap_source / "bootstrap_model.pt"
        embed_path = bootstrap_source / "bootstrap_embeddings.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Bootstrap model not found: {model_path}")
        if not embed_path.exists():
            raise FileNotFoundError(f"Bootstrap embeddings not found: {embed_path}")

        print("\n" + "=" * 60)
        print("Phase 1: Loading Bootstrap from Previous Run")
        print(f"  Source: {bootstrap_source}")
        print("=" * 60)

        # Warn if the source config used different bootstrap hyper-parameters
        source_config_path = bootstrap_source / "config.yaml"
        if source_config_path.exists():
            with open(source_config_path, "r") as f:
                source_cfg = yaml.safe_load(f)
            for key in ("bootstrap_frames", "bootstrap_epochs", "bootstrap_lr",
                        "bootstrap_batch_size", "min_box_area", "trainable_backbone_layers"):
                src_val = source_cfg.get(key)
                cur_val = getattr(config, key, None)
                if src_val is not None and cur_val is not None and src_val != cur_val:
                    print(f"  WARNING: {key} differs: source={src_val}, current={cur_val}")

        model = Detector(
            num_classes=config.num_classes,
            trainable_backbone_layers=config.trainable_backbone_layers,
            image_min_size=config.image_min_size,
            image_max_size=config.image_max_size,
            pretrained_backbone=config.pretrained_backbone,
            pretrained_detector=config.pretrained_detector,
        )
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        print(model)
        print(f"Loaded bootstrap model from {model_path.name}")

        embed_data = torch.load(embed_path, map_location="cpu")
        embedding_mean = embed_data["mean"]
        embedding_cov = embed_data["cov"]
        print(f"Loaded embeddings: mean {embedding_mean.shape}, cov {embedding_cov.shape}")

        # Copy artifacts into this run for provenance
        shutil.copy(model_path, run_dir / "bootstrap_model.pt")
        shutil.copy(embed_path, run_dir / "bootstrap_embeddings.pt")
        (run_dir / "bootstrap_source.txt").write_text(str(bootstrap_source))

    else:
        # ----- Run bootstrap from scratch -----
        print("\n" + "=" * 60)
        print("Phase 1: Bootstrap Training")
        print("=" * 60)

        bootstrap_dataset = DetectionDataset(
            manifest_path=manifest_path,
            split="train",
            transform=train_transform,
            augmentation=train_augmentation,
            frame_range=(0, config.bootstrap_frames),
            min_box_area=config.min_box_area,
        )

        bootstrap_loader = torch.utils.data.DataLoader(
            bootstrap_dataset,
            batch_size=config.bootstrap_batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=detection_collate,
        )

        model = Detector(
            num_classes=config.num_classes,
            trainable_backbone_layers=config.trainable_backbone_layers,
            image_min_size=config.image_min_size,
            image_max_size=config.image_max_size,
            pretrained_backbone=config.pretrained_backbone,
            pretrained_detector=config.pretrained_detector,
        )

        if config.load_checkpoint:
            checkpoint_path = PROJECT_ROOT / config.load_checkpoint
            print(f"Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])

        model = model.to(device)
        print(model)

        bootstrap_optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.bootstrap_lr,
            weight_decay=config.weight_decay,
        )

        print(f"\nBootstrap: {config.bootstrap_frames} frames, {config.bootstrap_epochs} epochs")
        final_loss, total_steps = bootstrap_train(
            model=model,
            train_loader=bootstrap_loader,
            optimizer=bootstrap_optimizer,
            device=device,
            epochs=config.bootstrap_epochs,
            max_grad_norm=config.max_grad_norm,
        )
        print(f"Bootstrap complete: final_loss={final_loss:.4f}, steps={total_steps}")

        print("\nRunning bootstrap smoke-check...")
        smoke_val_stream = DetectionStream(
            manifest_path=manifest_path,
            split="val",
            transform=val_transform,
            min_box_area=config.min_box_area,
            frame_range=(0, config.bootstrap_smoke_check_frames),
            verbose=False,
        )
        smoke_metrics = evaluate_detection(
            model,
            smoke_val_stream,
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
                "Bootstrap smoke-check failed: "
                f"mAP@50={smoke_metrics.get('mAP_50', 0.0):.4f} < "
                f"{config.bootstrap_smoke_min_map50:.4f}. "
                "Increase bootstrap strength or adjust detector initialization."
            )

        # Save bootstrap checkpoint
        torch.save(
            {"model_state_dict": model.state_dict()},
            run_dir / "bootstrap_model.pt",
        )

        # Collect embedding statistics
        print("\nCollecting embedding statistics...")
        embed_loader = torch.utils.data.DataLoader(
            bootstrap_dataset,
            batch_size=config.bootstrap_batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=detection_collate,
        )
        embedding_mean, embedding_cov = collect_embeddings(model, embed_loader, device)
        print(f"Embedding mean shape: {embedding_mean.shape}")
        print(f"Embedding cov shape:  {embedding_cov.shape}")

        # Save embeddings
        torch.save(
            {"mean": embedding_mean, "cov": embedding_cov},
            run_dir / "bootstrap_embeddings.pt",
        )

    # =========================================================================
    # Phase 2: Streaming
    # =========================================================================

    print("\n" + "=" * 60)
    print("Phase 2: Streaming Training")
    print("=" * 60)

    # Stream starts after bootstrap frames
    train_stream = DetectionStream(
        manifest_path=manifest_path,
        split="train",
        transform=train_transform,
        augmentation=train_augmentation,
        min_box_area=config.min_box_area,
        frame_range=(config.bootstrap_frames, None),
    )

    val_stream = DetectionStream(
        manifest_path=manifest_path,
        split="val",
        transform=val_transform,
        min_box_area=config.min_box_area,
    )

    # Streaming optimizer (fresh, not carrying bootstrap momentum)
    streaming_optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.streaming_lr,
        weight_decay=config.weight_decay,
    )

    # Filter policy
    filter_policy = create_filter_policy(
        config,
        bootstrap_mean=embedding_mean,
        bootstrap_cov=embedding_cov,
    )
    print(f"Filter policy: {config.filter_policy}")

    # Training buffer
    training_buffer = TrainingBuffer(capacity=config.buffer_capacity)
    print(f"Training buffer capacity: {config.buffer_capacity}")

    # Novelty tracker
    novelty_tracker = NoveltyTracker()

    # Metrics logger
    metrics_logger = StreamingMetricsLogger(
        log_dir=run_dir,
        checkpoint_interval=config.checkpoint_interval,
    )

    # Evaluation callback
    eval_interval = max(1, config.eval_every_n_items // config.checkpoint_interval)

    def eval_fn(m: nn.Module) -> dict:
        return evaluate_detection(m, val_stream, device, score_threshold=config.score_threshold)

    # Save run info
    dataset_info = {
        "bootstrap_frames": config.bootstrap_frames,
        "stream_frames": len(train_stream),
        "val_frames": len(val_stream),
        "bootstrap_reused": bootstrap_source is not None,
    }
    if bootstrap_source is not None:
        dataset_info["bootstrap_source"] = str(bootstrap_source)

    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        dataset_info=dataset_info,
        repo_path=PROJECT_ROOT,
    )

    # Run streaming training
    print(f"\nStreaming {len(train_stream)} frames...")
    result = train_on_stream(
        model=model,
        stream=train_stream,
        optimizer=streaming_optimizer,
        filter_policy=filter_policy,
        training_buffer=training_buffer,
        device=device,
        max_grad_norm=config.max_grad_norm,
        train_steps_per_buffer=config.train_steps_per_buffer,
        metrics_logger=metrics_logger,
        eval_fn=eval_fn,
        eval_every_n_checkpoints=eval_interval,
        novelty_tracker=novelty_tracker,
        total_items=len(train_stream),
    )

    print("\n" + "=" * 60)
    print("Streaming complete!")
    print(f"  Items processed : {result.items_processed}")
    print(f"  Accepted        : {result.items_accepted}")
    print(f"  Rejected        : {result.items_rejected}")
    print(f"  Buffer flushes  : {result.buffer_flushes}")
    print(f"  Optimizer steps : {result.optimizer_steps}")
    print("=" * 60)

    # Novelty summary
    novelty_tracker.print_summary()

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_detection(
        model, val_stream, device, score_threshold=config.score_threshold
    )
    print(f"  mAP:    {final_metrics['mAP']:.4f}")
    print(f"  mAP@50: {final_metrics.get('mAP_50', 0.0):.4f}")
    print(f"  mAP@75: {final_metrics.get('mAP_75', 0.0):.4f}")

    metrics_logger.print_summary()

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "final_metrics": final_metrics,
            "novelty_stats": novelty_tracker.get_stats(),
        },
        run_dir / "final_model.pt",
    )

    # Save final run info
    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        end_time=datetime.now(),
        best_metric=final_metrics["mAP"],
        best_metric_name="final_val_mAP",
        dataset_info=dataset_info,
        repo_path=PROJECT_ROOT,
    )

    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming detection experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--bootstrap-run-dir",
        type=str,
        default=None,
        help="Reuse bootstrap model + embeddings from this run directory "
             "(skips Phase 1). Overrides bootstrap_run_dir in config.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    command = " ".join(sys.argv)
    config = StreamingDetectionConfig.from_yaml(config_path)

    if args.bootstrap_run_dir:
        config.bootstrap_run_dir = args.bootstrap_run_dir

    main(config, config_path, command)
