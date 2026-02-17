"""
Evaluation utilities for binary classification models.

Metrics:
    compute_classification_metrics   Accuracy, precision, recall, F1

Evaluation functions:
    evaluate_offline_classification      Batch evaluation with DataLoader
    evaluate_streaming_classification    Online evaluation from temporal streams
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..core import StreamingDataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        preds: Predicted probabilities (after sigmoid), shape (N,).
        targets: Ground truth labels (0 or 1), shape (N,).
        threshold: Classification threshold for converting probabilities to labels.

    Returns:
        Dict with accuracy, precision, recall, f1, tp, fp, fn, tn.
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


# ---------------------------------------------------------------------------
# Offline (batch) evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_offline_classification(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval",
    metrics_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]] = compute_classification_metrics,
) -> Dict[str, float]:
    """
    Evaluate a classification model on a DataLoader (offline / batch).

    Args:
        model: PyTorch model to evaluate.
        loader: DataLoader for the evaluation dataset.
        criterion: Loss function.
        device: Device to run evaluation on.
        desc: Description for the progress bar.
        metrics_fn: Function to compute metrics from predictions and targets.
            Defaults to binary classification metrics.

    Returns:
        Dict with loss and all metrics returned by metrics_fn.
    """
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
    metrics = metrics_fn(all_preds, all_targets)
    metrics["loss"] = total_loss / max(num_batches, 1)

    return metrics


# ---------------------------------------------------------------------------
# Streaming (online) evaluation
# ---------------------------------------------------------------------------


def evaluate_streaming_classification(
    model: nn.Module,
    val_stream: StreamingDataset,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate a classification model on a validation stream (online).

    Args:
        model: Model to evaluate.
        val_stream: Streaming dataset for validation.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        Dict with loss, accuracy, precision, recall, f1.
    """
    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_items = 0

    with torch.no_grad():
        for stream_item in val_stream:
            image = stream_item.image.unsqueeze(0).to(device)
            target = torch.tensor([stream_item.target], dtype=torch.float32, device=device)

            logits = model(image)
            loss = criterion(logits, target)

            total_loss += loss.item()
            num_items += 1

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_targets.append(target.cpu())

    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_classification_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / max(num_items, 1)

    return metrics
