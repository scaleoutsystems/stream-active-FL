"""Evaluation methods and metrics.

Detection:
    evaluate_detection          COCO-style mAP evaluation
    COCO_IOU_THRESHOLDS         Standard IoU thresholds [0.50 : 0.05 : 0.95]
    DEFAULT_DOMAIN_DIMS         Marginal metadata keys for per-domain AP
    EXTENDED_DOMAIN_DIMS        DEFAULT_DOMAIN_DIMS + stream_block
    DETECTION_LABEL_TO_NAME     Model label -> class name (1-indexed)

Stream-block helpers:
    STREAM_BLOCK_DIM            Name of the joint (manifest) domain dim
    attach_stream_blocks        Enrich a frame_id -> metadata map with
                                stream_block labels matching the manifest
                                ordering strategy
    get_block_labeler           Lookup per-frame labeler for a strategy
"""

from .detection import (
    COCO_IOU_THRESHOLDS,
    DEFAULT_DOMAIN_DIMS,
    DETECTION_LABEL_TO_NAME,
    evaluate_detection,
)
from .stream_blocks import (
    STREAM_BLOCK_DIM,
    attach_stream_blocks,
    cityday_curated_block_label,
    get_block_labeler,
)

# Extended dims = marginal axes + joint manifest block.  Callers that
# attach stream_block labels should pass this to evaluate_detection.
EXTENDED_DOMAIN_DIMS = list(DEFAULT_DOMAIN_DIMS) + [STREAM_BLOCK_DIM]

__all__ = [
    "COCO_IOU_THRESHOLDS",
    "DEFAULT_DOMAIN_DIMS",
    "EXTENDED_DOMAIN_DIMS",
    "DETECTION_LABEL_TO_NAME",
    "STREAM_BLOCK_DIM",
    "attach_stream_blocks",
    "cityday_curated_block_label",
    "evaluate_detection",
    "get_block_labeler",
]
