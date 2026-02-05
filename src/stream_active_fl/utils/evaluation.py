"""Evaluation utilities for model assessment."""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from stream_active_fl.utils.training import compute_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval",
    metrics_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]] = compute_metrics,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.

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
