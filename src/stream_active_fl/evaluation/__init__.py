"""
Evaluation methods and metrics.

Classification (classification.py):
    compute_classification_metrics      Binary metrics (accuracy, precision, recall, F1)
    evaluate_offline_classification     Batch evaluation with DataLoader
    evaluate_streaming_classification   Online evaluation from temporal streams

Detection (detection.py):
    evaluate_detection                  COCO-style mAP evaluation for object detection
    COCO_IOU_THRESHOLDS                 Standard IoU thresholds [0.50 : 0.05 : 0.95]
    DETECTION_LABEL_TO_NAME             Model label → class name (1-indexed)
"""

from .classification import (
    compute_classification_metrics,
    evaluate_offline_classification,
    evaluate_streaming_classification,
)
from .detection import (
    COCO_IOU_THRESHOLDS,
    DETECTION_LABEL_TO_NAME,
    evaluate_detection,
)

__all__ = [
    "compute_classification_metrics",
    "evaluate_offline_classification",
    "evaluate_streaming_classification",
    "evaluate_detection",
    "COCO_IOU_THRESHOLDS",
    "DETECTION_LABEL_TO_NAME",
]
