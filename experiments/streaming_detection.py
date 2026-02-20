"""
Streaming 2D object detection experiment.

Trains an FCOS detector on ZOD sequences in strict temporal order with optional
filtering and replay. Parallel to streaming_classification.py but for detection.

Usage:
    python experiments/streaming_detection.py --config configs/streaming_detection_no_filter.yaml
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

# Suppress NVML warning (cosmetic, GPU still works fine)
warnings.filterwarnings("ignore", message="Can't initialize NVML")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from stream_active_fl.core import StreamingDataset, get_detection_augmentation, get_detection_transforms
from stream_active_fl.evaluation import evaluate_detection
from stream_active_fl.logging import StreamingMetricsLogger, create_run_dir, save_run_info
from stream_active_fl.memory import ReplayBuffer
from stream_active_fl.models import Detector
from stream_active_fl.policies import create_filter_policy
from stream_active_fl.training import train_on_detection_stream
from stream_active_fl.utils import set_seed


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StreamingDetectionConfig:
    """Streaming detection experiment configuration."""

    # Paths
    dataset_root: str = "/mnt/pr_2018_scaleout_workdir/ZOD256/ZOD_640x360"
    annotations_dir: str = "data/annotations_640x360"
    output_dir: str = "outputs/streaming_detection"

    # Dataset
    min_score: float = 0.5
    min_box_area: float = 0.0
    subsample_steps: int = 1

    # Model
    num_classes: int = 4  # 3 categories (person, car, traffic_light) + background
    trainable_backbone_layers: int = 0
    image_min_size: int = 360
    image_max_size: int = 640
    pretrained_backbone: bool = True

    # Load from checkpoint (optional)
    load_checkpoint: Optional[str] = None

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.0
    accumulation_steps: int = 1

    # Augmentation
    augment: bool = False
    hflip_prob: float = 0.5
    color_jitter: bool = True

    # Filtering policy
    filter_policy: Literal["none", "difficulty", "topk"] = "none"

    # Teacher confidence gate (applied before any filter policy)
    tau_teacher: float = 0.0
    store_gated: bool = False

    # Difficulty-based policy parameters
    adaptive: bool = True
    train_fraction: float = 0.3
    loss_window_size: int = 500
    warmup_items: int = 200
    tau_loss: float = 0.5
    store_skipped: bool = False

    # TopK policy parameters
    topk_window_size: int = 100
    topk_k: int = 30

    # Replay buffer
    use_replay: bool = False
    replay_capacity: int = 1000
    replay_batch_size: int = 16
    replay_admission_policy: Literal["fifo", "random", "reservoir"] = "fifo"
    replay_weight: float = 0.5

    # Evaluation
    eval_every_n_items: int = 5000
    checkpoint_interval: int = 1000
    score_threshold: float = 0.3  # minimum prediction score during evaluation

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StreamingDetectionConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Main
# =============================================================================

def main(config: StreamingDetectionConfig, config_path: Path, command: str) -> None:
    """Run streaming detection training."""

    start_time = datetime.now()

    print("=" * 60)
    print("Streaming Detection Training")
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

    # Copy config file
    shutil.copy(config_path, run_dir / "config.yaml")

    # Detection transforms (just ToTensor, model handles normalization)
    train_transform, val_transform = get_detection_transforms()

    # Augmentation for training (applied before ToTensor)
    train_augmentation = None
    if config.augment:
        train_augmentation = get_detection_augmentation(
            hflip_prob=config.hflip_prob,
            color_jitter=config.color_jitter,
        )
        print(f"\nAugmentation: hflip_prob={config.hflip_prob}, color_jitter={config.color_jitter}")

    # Create streaming datasets in detection mode
    print("\nLoading streaming datasets (detection mode)...")
    train_stream = StreamingDataset(
        dataset_root=config.dataset_root,
        annotations_dir=annotations_dir,
        split="train",
        transform=train_transform,
        min_score=config.min_score,
        min_box_area=config.min_box_area,
        subsample_steps=config.subsample_steps,
        task="detection",
        augmentation=train_augmentation,
        verbose=True,
    )

    val_stream = StreamingDataset(
        dataset_root=config.dataset_root,
        annotations_dir=annotations_dir,
        split="val",
        transform=val_transform,
        min_score=config.min_score,
        min_box_area=config.min_box_area,
        subsample_steps=config.subsample_steps,
        task="detection",
        verbose=True,
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

    # Load checkpoint if specified
    if config.load_checkpoint:
        checkpoint_path = PROJECT_ROOT / config.load_checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    print(model)

    # Optimizer (only trainable parameters)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Filter policy
    print("\nCreating filter policy...")
    filter_policy = create_filter_policy(config)
    print(f"  Policy: {config.filter_policy}")
    if config.filter_policy == "difficulty":
        if config.adaptive:
            print(f"  Mode: adaptive (train_fraction={config.train_fraction})")
            print(f"  Loss window: {config.loss_window_size}, warmup: {config.warmup_items}")
        else:
            print(f"  Mode: fixed (tau_loss={config.tau_loss})")
    elif config.filter_policy == "topk":
        print(f"  window_size: {config.topk_window_size}, k: {config.topk_k}")
    if config.tau_teacher > 0.0:
        print(f"  Teacher confidence gate: tau_teacher={config.tau_teacher}")

    # Training settings
    print(f"\nTraining settings:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max grad norm: {config.max_grad_norm}")
    print(f"  Accumulation steps: {config.accumulation_steps}")

    # Replay buffer
    replay_buffer = None
    if config.use_replay:
        print("\nCreating replay buffer...")
        replay_buffer = ReplayBuffer(
            capacity=config.replay_capacity,
            admission_policy=config.replay_admission_policy,
            device="cpu",
        )
        print(f"  Capacity: {config.replay_capacity}")
        print(f"  Admission policy: {config.replay_admission_policy}")
        print(f"  Replay batch size: {config.replay_batch_size}")
        print(f"  Replay weight: {config.replay_weight} (current item: {1.0 - config.replay_weight})")

    # Metrics logger (detection mode)
    metrics_logger = StreamingMetricsLogger(
        log_dir=run_dir,
        checkpoint_interval=config.checkpoint_interval,
        task="detection",
    )

    # Save initial run info
    dataset_info = {
        "train_total": len(train_stream),
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

    # Build evaluation callback
    eval_interval = max(1, config.eval_every_n_items // config.checkpoint_interval)

    def eval_fn(m: nn.Module) -> dict:
        return evaluate_detection(m, val_stream, device, score_threshold=config.score_threshold)

    # Run streaming detection training
    print("\n" + "=" * 60)
    print("Starting streaming detection training...")
    print(f"  Gradient clipping: max_norm={config.max_grad_norm}")
    print(f"  Accumulation steps: {config.accumulation_steps}")
    print("=" * 60)

    result = train_on_detection_stream(
        model=model,
        stream=train_stream,
        optimizer=optimizer,
        filter_policy=filter_policy,
        device=device,
        replay_buffer=replay_buffer,
        replay_batch_size=config.replay_batch_size,
        replay_weight=config.replay_weight,
        max_grad_norm=config.max_grad_norm,
        accumulation_steps=config.accumulation_steps,
        filter_computes_forward=(config.filter_policy != "none"),
        metrics_logger=metrics_logger,
        eval_fn=eval_fn,
        eval_every_n_checkpoints=eval_interval,
        total_items=len(train_stream),
    )

    print("\n" + "=" * 60)
    print("Streaming detection training complete!")
    print(f"  Items processed: {result.items_processed}")
    print(f"  Items trained:   {result.items_trained}")
    print(f"  Optimizer steps: {result.optimizer_steps}")
    print("=" * 60)

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_detection(
        model, val_stream, device, score_threshold=config.score_threshold
    )
    print(f"  mAP (COCO):       {final_metrics['mAP']:.4f}")
    print(f"  mAP@50:           {final_metrics.get('mAP_50', 0.0):.4f}")
    print(f"  mAP@75:           {final_metrics.get('mAP_75', 0.0):.4f}")
    print(f"  AP_person:        {final_metrics.get('AP_person', 0.0):.4f}")
    print(f"  AP_car:           {final_metrics.get('AP_car', 0.0):.4f}")
    print(f"  AP_traffic_light: {final_metrics.get('AP_traffic_light', 0.0):.4f}")

    # Print metrics summary
    metrics_logger.print_summary()

    # Save final model
    final_path = run_dir / "final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
            "final_metrics": final_metrics,
        },
        final_path,
    )

    # Save final run info
    end_time = datetime.now()
    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        end_time=end_time,
        best_metric=final_metrics["mAP"],
        best_metric_name="final_val_mAP",
        dataset_info=dataset_info,
        repo_path=PROJECT_ROOT,
    )

    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming detection training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    command = " ".join(sys.argv)

    config = StreamingDetectionConfig.from_yaml(config_path)
    main(config, config_path, command)
