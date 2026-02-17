"""
Streaming baseline training experiment.

Trains a classifier on ZOD sequences in strict temporal order with optional
filtering and replay. This is the core streaming learning pipeline.

Usage:
    python experiments/streaming_baseline.py --config configs/streaming_no_filter.yaml
    python experiments/streaming_baseline.py --config configs/streaming_difficulty_adaptive.yaml
"""

from __future__ import annotations

import argparse
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
from tqdm import tqdm
import yaml

# Suppress NVML warning (cosmetic, GPU still works fine)
warnings.filterwarnings("ignore", message="Can't initialize NVML")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from stream_active_fl.core import StreamingDataset, get_classification_transforms
from stream_active_fl.evaluation import evaluate_streaming_classification
from stream_active_fl.logging import StreamingMetricsLogger, create_run_dir, save_run_info
from stream_active_fl.memory import ReplayBuffer
from stream_active_fl.models import Classifier
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
class Config:
    """Streaming experiment configuration."""

    # Paths
    dataset_root: str = "/mnt/pr_2018_scaleout_workdir/ZOD256/ZOD_640x360"
    annotations_dir: str = "data/annotations_640x360"
    output_dir: str = "outputs/streaming_baseline"

    # Dataset
    target_category: int = 0
    min_score: float = 0.5
    subsample_steps: int = 1

    # Model
    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = True
    dropout: float = 0.0
    
    # Load from pretrained offline baseline (optional)
    load_checkpoint: Optional[str] = None

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.0
    accumulation_steps: int = 1

    # Class imbalance
    pos_weight: str | float = "auto"

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
    replay_batch_size: int = 32
    replay_admission_policy: Literal["fifo", "random", "reservoir"] = "fifo"
    replay_weight: float = 0.5  # Weight for replay loss (current item gets 1 - replay_weight)

    # Evaluation
    eval_every_n_items: int = 1000
    checkpoint_interval: int = 1000

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Training utilities
# =============================================================================

def compute_pos_weight_from_streaming_dataset(dataset: StreamingDataset) -> float:
    """Compute pos_weight from streaming dataset metadata."""
    num_positive = 0
    total = 0
    for seq_meta in dataset.sequence_metadata:
        frame_info = seq_meta["frame_info"]
        for frame_idx in range(0, seq_meta["num_frames"], dataset.subsample_steps):
            info = frame_info.get(frame_idx, {"has_target": False})
            if info["has_target"]:
                num_positive += 1
            total += 1
    
    num_negative = total - num_positive
    if num_positive == 0:
        return 1.0
    return num_negative / num_positive


def create_filter_policy(config: Config) -> FilterPolicy:
    """Create filter policy from config, optionally wrapped with teacher gate."""
    # Build the inner policy
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

    # Optionally wrap with teacher confidence gate
    if config.tau_teacher > 0.0:
        policy = TeacherConfidenceGate(
            inner_policy=policy,
            tau_teacher=config.tau_teacher,
            store_gated=config.store_gated,
        )

    return policy


def perform_update(
    model: nn.Module,
    criterion: nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    replay_batch: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    replay_weight: float = 0.5,
    accumulation_steps: int = 1,
) -> float:
    """
    Compute loss and accumulate gradients (backward pass only).

    The caller is responsible for optimizer.zero_grad(), gradient clipping,
    and optimizer.step(). This separation enables gradient accumulation
    across multiple stream items.

    Computes separate losses for the current item and replay batch, then
    combines them with explicit weighting. This prevents the replay batch
    (typically 32 samples) from drowning out the current item's gradient.

    Args:
        model: The model to update.
        criterion: Loss function.
        image: Current stream item image tensor.
        target: Current stream item target tensor.
        replay_batch: Optional dict with "image" and "target" tensors.
        device: Device to run on.
        replay_weight: Weight for replay loss. Current item loss gets
            weight (1 - replay_weight). Default 0.5 gives equal weight.
        accumulation_steps: Number of items over which gradients are accumulated.
            Loss is divided by this value so accumulated gradient magnitude
            matches a single-step update.

    Returns:
        Unscaled combined loss value (for logging).
    """
    model.train()

    # Current item loss
    image = image.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)
    logits_current = model(image)
    loss_current = criterion(logits_current, target)

    if replay_batch is not None:
        # Replay loss (computed separately)
        replay_images = replay_batch["image"].to(device)
        replay_targets = replay_batch["target"].to(device)
        logits_replay = model(replay_images)
        loss_replay = criterion(logits_replay, replay_targets)

        # Weighted combination: equal voice for current item and replay
        loss = (1.0 - replay_weight) * loss_current + replay_weight * loss_replay
    else:
        loss = loss_current

    # Scale loss for gradient accumulation and backward
    (loss / accumulation_steps).backward()

    return loss.item()


# =============================================================================
# Main streaming loop
# =============================================================================

def streaming_train(
    model: nn.Module,
    train_stream: StreamingDataset,
    val_stream: StreamingDataset,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    filter_policy: FilterPolicy,
    replay_buffer: Optional[ReplayBuffer],
    metrics_logger: StreamingMetricsLogger,
    device: torch.device,
    config: Config,
) -> None:
    """
    Main streaming training loop.

    Processes train_stream in temporal order, applying filter policy and
    optionally using replay. Periodically evaluates on val_stream.
    """
    print("\n" + "=" * 60)
    print("Starting streaming training...")
    print(f"  Gradient clipping: max_norm={config.max_grad_norm}")
    print(f"  Accumulation steps: {config.accumulation_steps}")
    print("=" * 60)

    checkpoint_idx = 0
    accumulation_steps = config.accumulation_steps
    train_count = 0  # train items since last optimizer step

    # Initialize gradient state for accumulation
    optimizer.zero_grad()

    # Create progress bar
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
                replay_batch = replay_buffer.sample(config.replay_batch_size, device=str(device))

            # Compute loss and accumulate gradients
            loss = perform_update(
                model,
                criterion,
                stream_item.image,
                torch.tensor(stream_item.target, dtype=torch.float32),
                replay_batch,
                device,
                replay_weight=config.replay_weight,
                accumulation_steps=accumulation_steps,
            )

            train_count += 1

            # Step optimizer after accumulating enough gradients
            if train_count % accumulation_steps == 0:
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        config.max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad()

        # Offer every item to the replay buffer (regardless of filter action)
        # so the buffer stays representative of the full stream distribution.
        # The buffer's admission policy (e.g. reservoir sampling) decides
        # whether to actually store or reject each item.
        if replay_buffer is not None:
            replay_buffer.add(stream_item.to_dict())

        # Checkpoint and evaluation
        if metrics_logger.should_checkpoint():
            checkpoint_idx += 1

            # Log checkpoint metrics
            buffer_stats = replay_buffer.get_stats() if replay_buffer else None
            filter_stats = filter_policy.get_stats()
            metrics_logger.log_checkpoint(checkpoint_idx, buffer_stats, filter_stats)

            # Log per-class filter selection stats (interval since last checkpoint)
            selection_stats = filter_policy.get_selection_stats()
            metrics_logger.log_filter_stats(checkpoint_idx, selection_stats)
            filter_policy.reset_selection_stats()

            # Evaluate on validation stream
            eval_interval = max(1, config.eval_every_n_items // config.checkpoint_interval)
            if checkpoint_idx % eval_interval == 0:
                eval_metrics = evaluate_streaming_classification(model, val_stream, criterion, device)
                metrics_logger.log_evaluation(checkpoint_idx, eval_metrics)

                # Update progress bar
                pbar.set_postfix({
                    "val_f1": f"{eval_metrics['f1']:.3f}",
                    "train_rate": f"{filter_stats.get('train_rate', 1.0):.3f}",
                })

    # Flush any remaining accumulated gradients
    if train_count % accumulation_steps != 0:
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.max_grad_norm,
            )
        optimizer.step()
        optimizer.zero_grad()

    print("\n" + "=" * 60)
    print("Streaming training complete!")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main(config: Config, config_path: Path, command: str) -> None:
    """Run streaming baseline training."""

    start_time = datetime.now()

    print("=" * 60)
    print("Streaming Baseline Training")
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

    # Transforms
    train_transform, val_transform = get_classification_transforms()

    # Create streaming datasets
    print("\nLoading streaming datasets...")
    train_stream = StreamingDataset(
        dataset_root=config.dataset_root,
        annotations_dir=annotations_dir,
        split="train",
        transform=train_transform,
        target_category=config.target_category,
        min_score=config.min_score,
        subsample_steps=config.subsample_steps,
        verbose=True,
    )

    val_stream = StreamingDataset(
        dataset_root=config.dataset_root,
        annotations_dir=annotations_dir,
        split="val",
        transform=val_transform,
        target_category=config.target_category,
        min_score=config.min_score,
        subsample_steps=config.subsample_steps,
        verbose=True,
    )

    # Model
    print("\nInitializing model...")
    model = Classifier(
        backbone=config.backbone,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        dropout=config.dropout,
    )
    
    # Load checkpoint if specified
    if config.load_checkpoint:
        checkpoint_path = PROJECT_ROOT / config.load_checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    print(model)

    # Loss function
    if config.pos_weight == "auto":
        pos_weight_value = compute_pos_weight_from_streaming_dataset(train_stream)
        print(f"\nAuto-computed pos_weight: {pos_weight_value:.2f}")
    else:
        pos_weight_value = float(config.pos_weight)

    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
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
        print(f"  Teacher confidence gate: tau_teacher={config.tau_teacher} (positive items only)")

    # Replay buffer
    replay_buffer = None
    if config.use_replay:
        print("\nCreating replay buffer...")
        replay_buffer = ReplayBuffer(
            capacity=config.replay_capacity,
            admission_policy=config.replay_admission_policy,
            device="cpu",  # Store on CPU to save GPU memory
        )
        print(f"  Capacity: {config.replay_capacity}")
        print(f"  Admission policy: {config.replay_admission_policy}")
        print(f"  Replay batch size: {config.replay_batch_size}")
        print(f"  Replay weight: {config.replay_weight} (current item: {1.0 - config.replay_weight})")

    # Metrics logger
    metrics_logger = StreamingMetricsLogger(
        log_dir=run_dir,
        checkpoint_interval=config.checkpoint_interval,
    )

    # Save initial run info
    dataset_info = {
        "train_total": len(train_stream),
        "val_total": len(val_stream),
    }
    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        dataset_info=dataset_info,
        repo_path=PROJECT_ROOT,
    )

    # Run streaming training
    streaming_train(
        model=model,
        train_stream=train_stream,
        val_stream=val_stream,
        criterion=criterion,
        optimizer=optimizer,
        filter_policy=filter_policy,
        replay_buffer=replay_buffer,
        metrics_logger=metrics_logger,
        device=device,
        config=config,
    )

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_streaming_classification(model, val_stream, criterion, device)
    print(f"  Val Loss: {final_metrics['loss']:.4f}")
    print(f"  Val Acc:  {final_metrics['accuracy']:.4f}")
    print(f"  Val F1:   {final_metrics['f1']:.4f}")

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
        best_metric=final_metrics["f1"],
        best_metric_name="final_val_f1",
        dataset_info=dataset_info,
        repo_path=PROJECT_ROOT,
    )

    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming baseline training")
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

    # Reconstruct command for logging
    command = " ".join(sys.argv)

    config = Config.from_yaml(config_path)
    main(config, config_path, command)
