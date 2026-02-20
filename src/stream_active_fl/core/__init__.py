"""
Core abstractions for stream learning.

Data structures and dataset implementations for the ZOD (Zenseact Open
Dataset) autonomous-driving dataset, supporting both offline and streaming
training for classification and detection tasks.

Datasets:
    ZODClassificationDataset         Offline classification (shuffled, DataLoader-compatible)
    ZODDetectionDataset     Offline detection (shuffled, DataLoader-compatible)
    StreamingDataset        Online streaming for both tasks (temporal order, iterator)

Data:
    StreamItem              Single frame flowing through the streaming pipeline

Transforms & augmentation:
    get_classification_transforms      Classification transforms (Resize + Normalize)
    get_detection_transforms    Detection transforms (ToTensor only)
    DetectionAugmentation       Spatial + photometric augmentation for detection
    get_detection_augmentation  Factory for DetectionAugmentation

Collate functions:
    classification_collate       Classification batching (drops failed reads)
    detection_collate       Detection batching (variable-length annotations)

Constants:
    CATEGORY_ID_TO_NAME     {0: "person", 1: "car", 2: "traffic_light"}
    CATEGORY_NAME_TO_ID     Inverse mapping
"""

from .datasets import (
    CATEGORY_ID_TO_NAME,
    CATEGORY_NAME_TO_ID,
    DetectionAugmentation,
    ZODDetectionDataset,
    ZODClassificationDataset,
    StreamingDataset,
    classification_collate,
    detection_collate,
    get_classification_transforms,
    get_detection_augmentation,
    get_detection_transforms,
)
from .items import StreamItem
from .partitioning import ClientStream, partition_sequences

__all__ = [
    "CATEGORY_ID_TO_NAME",
    "CATEGORY_NAME_TO_ID",
    "ClientStream",
    "DetectionAugmentation",
    "ZODDetectionDataset",
    "ZODClassificationDataset",
    "StreamingDataset",
    "StreamItem",
    "classification_collate",
    "detection_collate",
    "get_classification_transforms",
    "get_detection_augmentation",
    "get_detection_transforms",
    "partition_sequences",
]
