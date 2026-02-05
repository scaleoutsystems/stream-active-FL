"""
Offline baseline training experiment.

Trains a binary classifier on the full ZOD dataset (shuffled, multi-epoch)
to establish an upper-bound baseline for comparison with streaming methods.

Usage:
    python experiments/offline_baseline.py --config configs/offline_baseline.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


def _worker_init_fn(worker_id: int) -> None:
    """Suppress warnings in DataLoader worker processes to keep tqdm clean."""
    warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from stream_active_fl.data import (
    ZODFrameDataset,
    collate_drop_none,
    get_default_transforms,
)
from stream_active_fl.models import Classifier


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Experiment configuration."""

    # Paths
    dataset_root: str = "/mnt/pr_2018_scaleout_workdir/ZODCropped"
    annotations_dir: str = "data/annotations_ZODCropped"
    output_dir: str = "outputs/offline_baseline"

    # Dataset
    target_category: int = 0
    min_score: float = 0.5
    subsample_steps: int = 1

    # Model
    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = True
    dropout: float = 0.0

    # Training
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Class imbalance
    pos_weight: str | float = "auto"

    # Evaluation
    eval_every_n_epochs: int = 1

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

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(dataset: ZODFrameDataset) -> float:
    """Compute pos_weight for BCEWithLogitsLoss based on class imbalance."""
    num_positive = sum(1 for _, _, t, _ in dataset.samples if t == 1.0)
    num_negative = len(dataset.samples) - num_positive
    if num_positive == 0:
        return 1.0
    return num_negative / num_positive


def compute_metrics(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        preds: Predicted probabilities (after sigmoid).
        targets: Ground truth labels (0 or 1).
        threshold: Classification threshold.

    Returns:
        Dict with accuracy, precision, recall, f1.
    """
    pred_labels = (preds >= threshold).float()

    tp = ((pred_labels == 1) & (targets == 1)).sum().item()
    fp = ((pred_labels == 1) & (targets == 0)).sum().item()
    fn = ((pred_labels == 0) & (targets == 1)).sum().item()
    tn = ((pred_labels == 0) & (targets == 0)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# =============================================================================
# Training and evaluation loops
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        if batch is None:
            continue

        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Collect predictions for metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_targets.append(targets.cpu())

        pbar.set_postfix(loss=loss.item())

    # Compute epoch metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / max(num_batches, 1)

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval",
) -> Dict[str, float]:
    """Evaluate on a dataset."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        if batch is None:
            continue

        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu())
        all_targets.append(targets.cpu())

    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / max(num_batches, 1)

    return metrics


# =============================================================================
# Main
# =============================================================================

def main(config: Config) -> None:
    """Run offline baseline training."""

    print("=" * 60)
    print("Offline Baseline Training")
    print("=" * 60)

    # Setup
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve paths
    annotations_dir = PROJECT_ROOT / config.annotations_dir
    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_transform, val_transform = get_default_transforms()

    # Datasets
    print("\nLoading datasets...")
    train_dataset = ZODFrameDataset(
        dataset_root=config.dataset_root,
        annotations_dir=annotations_dir,
        split="train",
        transform=train_transform,
        target_category=config.target_category,
        min_score=config.min_score,
        subsample_steps=config.subsample_steps,
        verbose=True,
    )

    val_dataset = ZODFrameDataset(
        dataset_root=config.dataset_root,
        annotations_dir=annotations_dir,
        split="val",
        transform=val_transform,
        target_category=config.target_category,
        min_score=config.min_score,
        subsample_steps=config.subsample_steps,
        verbose=True,
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_drop_none,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_drop_none,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    # Model
    print("\nInitializing model...")
    model = Classifier(
        backbone=config.backbone,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        dropout=config.dropout,
    )
    model = model.to(device)
    print(model)

    # Loss function with class weighting
    if config.pos_weight == "auto":
        pos_weight_value = compute_pos_weight(train_dataset)
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

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_f1 = 0.0
    history = {"train": [], "val": []}

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        history["train"].append(train_metrics)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{config.epochs} "
            f"[{epoch_time:.1f}s] "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Acc: {train_metrics['accuracy']:.4f} "
            f"F1: {train_metrics['f1']:.4f}"
        )

        # Validate
        if epoch % config.eval_every_n_epochs == 0:
            val_metrics = evaluate(
                model, val_loader, criterion, device, desc=f"Epoch {epoch} [Val]"
            )
            history["val"].append(val_metrics)

            print(
                f"         Val Loss: {val_metrics['loss']:.4f} "
                f"Acc: {val_metrics['accuracy']:.4f} "
                f"Prec: {val_metrics['precision']:.4f} "
                f"Rec: {val_metrics['recall']:.4f} "
                f"F1: {val_metrics['f1']:.4f}"
            )

            # Save best model
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                checkpoint_path = output_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "config": config.__dict__,
                    },
                    checkpoint_path,
                )
                print(f"         Saved best model (F1={best_val_f1:.4f})")

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config.__dict__,
        },
        final_path,
    )

    # Final summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline baseline training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/offline_baseline.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = Config.from_yaml(config_path)
    main(config)
