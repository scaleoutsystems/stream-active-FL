"""
Evaluation methods and metrics.

Detection:
    evaluate_detection          COCO-style mAP evaluation
    COCO_IOU_THRESHOLDS         Standard IoU thresholds [0.50 : 0.05 : 0.95]
    DETECTION_LABEL_TO_NAME     Model label -> class name (1-indexed)

Novelty:
    NoveltyTracker              Tracks novel-category acceptance rates
"""

from .detection import (
    COCO_IOU_THRESHOLDS,
    DETECTION_LABEL_TO_NAME,
    evaluate_detection,
)
from .novelty import NoveltyTracker

__all__ = [
    "COCO_IOU_THRESHOLDS",
    "DETECTION_LABEL_TO_NAME",
    "NoveltyTracker",
    "evaluate_detection",
]
