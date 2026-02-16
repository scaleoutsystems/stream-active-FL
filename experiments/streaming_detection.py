"""
Streaming 2D object detection experiment.

Trains an FCOS detector on ZOD sequences in strict temporal order with optional
filtering and replay. Parallel to streaming_baseline.py but for detection.

Usage:
    python experiments/streaming_detection.py --config configs/detection_streaming_no_filter.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

# Suppress NVML warning (cosmetic, GPU still works fine)
warnings.filterwarnings("ignore", message="Can't initialize NVML")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from stream_active_fl.core import StreamingDataset, get_detection_transforms
from stream_active_fl.core.items import StreamItem
from stream_active_fl.evaluation import evaluate_detection
from stream_active_fl.logging import StreamingMetricsLogger, create_run_dir, save_run_info
from stream_active_fl.memory import ReplayBuffer
from stream_active_fl.models import StreamingDetector
from stream_active_fl.policies import (
    DifficultyBasedPolicy,
    FilterPolicy,
    NoFilterPolicy,
    TeacherConfidenceGate,
    TopKPolicy,
)
from stream_active_fl.utils import set_seed


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DetectionConfig:
    """Streaming detection experiment configuration."""

    # Paths
    dataset_root: str = "/mnt/pr_2018_scaleout_workdir/ZOD256/ZOD_640x360"
    annotations_dir: str = "data/annotations_640x360"
    output_dir: str = "outputs/detection_streaming_no_filter"

    # Dataset
    min_score: float = 0.5
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
    def from_yaml(cls, path: str | Path) -> "DetectionConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Training utilities
# =============================================================================

def create_filter_policy(config: DetectionConfig) -> FilterPolicy:
    """Create filter policy from config, optionally wrapped with teacher gate."""
    if config.filter_policy == "none":
        policy = NoFilterPolicy()
    elif config.filter_policy == "difficulty":
        policy = DifficultyBasedPolicy(
            adaptive=config.adaptive,
            train_fraction=config.train_fraction,
            loss_window_size=config.loss_window_size,
            warmup_items=config.warmup_items,
            tau_loss=config.tau_loss,
            store_skipped=config.store_skipped,
        )
    elif config.filter_policy == "topk":
        policy = TopKPolicy(
            window_size=config.topk_window_size,
            k=config.topk_k,
        )
    else:
        raise ValueError(f"Unknown filter policy: {config.filter_policy}")

    if config.tau_teacher > 0.0:
        policy = TeacherConfidenceGate(
            inner_policy=policy,
            tau_teacher=config.tau_teacher,
            store_gated=config.store_gated,
        )

    return policy


def perform_detection_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stream_item: StreamItem,
    replay_batch: Optional[Dict[str, Any]],
    device: torch.device,
    replay_weight: float = 0.5,
) -> float:
    """
    Perform a single detection parameter update.

    The detection model computes its own loss internally (classification +
    bbox regression + centerness). Current item and replay batch get
    separate forward passes with explicit weighting.

    Args:
        model: Detection model (StreamingDetector).
        optimizer: Optimizer.
        stream_item: Current stream item with annotations.
        replay_batch: Optional dict with "images" and "targets" lists.
        device: Device to run on.
        replay_weight: Weight for replay loss. Current gets (1 - replay_weight).

    Returns:
        Combined loss value.
    """
    model.train()
    optimizer.zero_grad()

    # Current item loss
    image = stream_item.image.to(device)
    target = {
        "boxes": stream_item.annotations["boxes"].to(device),
        "labels": stream_item.annotations["labels"].to(device),
    }
    loss_dict = model([image], [target])
    loss_current = sum(loss_dict.values())

    if replay_batch is not None:
        # Replay loss (separate forward pass for proper weighting)
        replay_images = [img.to(device) for img in replay_batch["images"]]
        replay_targets = [
            {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
            for t in replay_batch["targets"]
        ]
        loss_dict_replay = model(replay_images, replay_targets)
        loss_replay = sum(loss_dict_replay.values())

        loss = (1.0 - replay_weight) * loss_current + replay_weight * loss_replay
    else:
        loss = loss_current

    loss.backward()
    optimizer.step()

    return loss.item()


# =============================================================================
# Main streaming loop
# =============================================================================

def streaming_detection_train(
    model: nn.Module,
    train_stream: StreamingDataset,
    val_stream: StreamingDataset,
    optimizer: torch.optim.Optimizer,
    filter_policy: FilterPolicy,
    replay_buffer: Optional[ReplayBuffer],
    metrics_logger: StreamingMetricsLogger,
    device: torch.device,
    config: DetectionConfig,
) -> None:
    """
    Main streaming detection training loop.

    Processes train_stream in temporal order, applying filter policy and
    optionally using replay. Periodically evaluates on val_stream with mAP.
    """
    print("\n" + "=" * 60)
    print("Starting streaming detection training...")
    print("=" * 60)

    checkpoint_idx = 0

    # criterion=None for detection (model computes its own loss)
    # but filter policies require it in their signature
    criterion = None

    pbar = tqdm(train_stream, desc="Processing stream", total=len(train_stream))

    for stream_item in pbar:
        # Select action using filter policy
        action = filter_policy.select_action(stream_item, model, criterion, device)

        # Log stream item processing
        forward_pass = (action == "train") or (config.filter_policy != "none")
        backward_pass = (action == "train")
        metrics_logger.log_stream_item(action, forward_pass, backward_pass)

        # Execute action
        if action == "train":
            # Sample replay batch if available
            replay_batch = None
            if replay_buffer is not None and len(replay_buffer) > 0:
                replay_batch = replay_buffer.sample(
                    config.replay_batch_size, device=str(device)
                )

            # Perform detection update
            loss = perform_detection_update(
                model,
                optimizer,
                stream_item,
                replay_batch,
                device,
                replay_weight=config.replay_weight,
            )

        # Offer every item to the replay buffer
        if replay_buffer is not None:
            replay_buffer.add(stream_item.to_dict())

        # Checkpoint and evaluation
        if metrics_logger.should_checkpoint():
            checkpoint_idx += 1

            # Log checkpoint metrics
            buffer_stats = replay_buffer.get_stats() if replay_buffer else None
            filter_stats = filter_policy.get_stats()
            metrics_logger.log_checkpoint(checkpoint_idx, buffer_stats, filter_stats)

            # Log per-class filter selection stats
            selection_stats = filter_policy.get_selection_stats()
            metrics_logger.log_filter_stats(checkpoint_idx, selection_stats)
            filter_policy.reset_selection_stats()

            # Evaluate on validation stream
            eval_interval = max(1, config.eval_every_n_items // config.checkpoint_interval)
            if checkpoint_idx % eval_interval == 0:
                eval_metrics = evaluate_detection(
                    model,
                    val_stream,
                    device,
                    score_threshold=config.score_threshold,
                )
                metrics_logger.log_evaluation(checkpoint_idx, eval_metrics)

                pbar.set_postfix({
                    "mAP": f"{eval_metrics['mAP']:.3f}",
                    "mAP50": f"{eval_metrics.get('mAP_50', 0.0):.3f}",
                    "train_rate": f"{filter_stats.get('train_rate', 1.0):.3f}",
                })

    print("\n" + "=" * 60)
    print("Streaming detection training complete!")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main(config: DetectionConfig, config_path: Path, command: str) -> None:
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

    # Create streaming datasets in detection mode
    print("\nLoading streaming datasets (detection mode)...")
    train_stream = StreamingDataset(
        dataset_root=config.dataset_root,
        annotations_dir=annotations_dir,
        split="train",
        transform=train_transform,
        min_score=config.min_score,
        subsample_steps=config.subsample_steps,
        task="detection",
        verbose=True,
    )

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

    # Model
    print("\nInitializing detection model...")
    model = StreamingDetector(
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

    # Run streaming detection training
    streaming_detection_train(
        model=model,
        train_stream=train_stream,
        val_stream=val_stream,
        optimizer=optimizer,
        filter_policy=filter_policy,
        replay_buffer=replay_buffer,
        metrics_logger=metrics_logger,
        device=device,
        config=config,
    )

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

    config = DetectionConfig.from_yaml(config_path)
    main(config, config_path, command)
