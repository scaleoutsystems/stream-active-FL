"""
Core abstractions for stream learning.

This module contains fundamental data structures and dataset implementations:
- ZODFrameDataset: Offline frame-level dataset with batch sampling
- StreamingDataset: Temporal sequence dataset for online learning
- StreamItem: Single item in a data stream
- Transforms and utilities for data loading
"""

from .datasets import (
    CATEGORY_ID_TO_NAME,
    CATEGORY_NAME_TO_ID,
    ZODFrameDataset,
    StreamingDataset,
    collate_drop_none,
    get_default_transforms,
)
from .items import StreamItem

__all__ = [
    "CATEGORY_ID_TO_NAME",
    "CATEGORY_NAME_TO_ID",
    "ZODFrameDataset",
    "StreamingDataset",
    "StreamItem",
    "collate_drop_none",
    "get_default_transforms",
]
