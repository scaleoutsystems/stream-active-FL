"""
Evaluation methods and metrics.

Evaluation utilities for different training paradigms:
- compute_metrics: Binary classification metrics (accuracy, precision, recall, F1)
- evaluate_offline: Batch evaluation with DataLoader
- evaluate_streaming: Online evaluation from temporal streams
"""

from .metrics import compute_metrics
from .offline import evaluate as evaluate_offline
from .streaming import evaluate_streaming

__all__ = [
    "compute_metrics",
    "evaluate_offline",
    "evaluate_streaming",
]
