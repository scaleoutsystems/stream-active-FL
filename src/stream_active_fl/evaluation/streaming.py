"""Evaluation for online streaming training."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..core import StreamingDataset
from .metrics import compute_metrics


def evaluate_streaming(
    model: nn.Module,
    val_stream: StreamingDataset,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on validation stream (without updating).

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
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / max(num_items, 1)

    return metrics
