"""
Evaluation methods and metrics.

Evaluation utilities for different training paradigms:
- compute_metrics: Binary classification metrics (accuracy, precision, recall, F1)
- evaluate_offline: Batch evaluation with DataLoader
- evaluate_streaming: Online evaluation from temporal streams
- evaluate_detection: COCO-style mAP evaluation for object detection
"""

from .metrics import compute_metrics
from .offline import evaluate as evaluate_offline
from .streaming import evaluate_streaming
from .detection import (
    COCO_IOU_THRESHOLDS,
    DETECTION_LABEL_TO_NAME,
    evaluate_detection,
)

__all__ = [
    "compute_metrics",
    "evaluate_offline",
    "evaluate_streaming",
    "evaluate_detection",
    "COCO_IOU_THRESHOLDS",
    "DETECTION_LABEL_TO_NAME",
]
