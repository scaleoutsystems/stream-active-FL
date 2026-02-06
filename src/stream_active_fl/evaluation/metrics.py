"""Metric computation functions for binary classification."""

from __future__ import annotations

from typing import Dict

import torch


def compute_metrics(
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
