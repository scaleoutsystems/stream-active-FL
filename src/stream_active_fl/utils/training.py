"""Training utilities for reproducibility and metrics computation."""

from __future__ import annotations

import random
import warnings
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy, PyTorch CPU and CUDA,
    and configures cuDNN for deterministic behavior.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker init function that suppresses warnings.

    Use as worker_init_fn parameter in DataLoader to keep tqdm output clean.

    Args:
        worker_id: Worker ID (unused, required by DataLoader interface).
    """
    warnings.filterwarnings("ignore")


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
