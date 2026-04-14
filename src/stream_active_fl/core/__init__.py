"""
Core abstractions for streaming active learning.

Datasets:
    DetectionDataset         Offline detection (shuffled, DataLoader-compatible)
    DetectionStream          Online streaming (chronological order, iterator)

Data:
    StreamItem               Single frame flowing through the streaming pipeline

Transforms & augmentation:
    get_detection_transforms    Detection transforms (ToTensor only)
    DetectionAugmentation       Spatial + photometric augmentation for detection
    get_detection_augmentation  Factory for DetectionAugmentation

Collate:
    detection_collate           Detection batching (variable-length annotations)

Constants:
    CATEGORY_ID_TO_NAME      {0: "Vehicle", 1: "VulnerableVehicle", ...}
    CATEGORY_NAME_TO_ID      Inverse mapping
    NUM_CLASSES              Number of classes including background
"""

from .datasets import (
    CATEGORY_ID_TO_NAME,
    CATEGORY_NAME_TO_ID,
    NUM_CLASSES,
    ClassMapping,
    DetectionAugmentation,
    DetectionDataset,
    DetectionStream,
    build_class_mapping,
    detection_collate,
    get_detection_augmentation,
    get_detection_transforms,
)
from .items import StreamItem
from .partitioning import partition_frames, partition_frames_by_domain

__all__ = [
    "CATEGORY_ID_TO_NAME",
    "CATEGORY_NAME_TO_ID",
    "NUM_CLASSES",
    "ClassMapping",
    "build_class_mapping",
    "partition_frames",
    "partition_frames_by_domain",
    "DetectionAugmentation",
    "DetectionDataset",
    "DetectionStream",
    "StreamItem",
    "detection_collate",
    "get_detection_augmentation",
    "get_detection_transforms",
]
