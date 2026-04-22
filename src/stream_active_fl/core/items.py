"""
Core data structures for stream learning.

Defines the StreamItem class, the fundamental unit of data flowing through
the streaming pipeline. Each StreamItem represents a single camera frame
with its associated detection annotations and metadata.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import torch


class StreamItem:
    """
    A single camera frame in the data stream.

    Carries detection annotations so the same StreamItem can be consumed by
    the bootstrap trainer, the streaming filter, and the training buffer.

    Attributes:
        image: Image tensor of shape (C, H, W).
        annotations: Detection targets with "boxes" (FloatTensor[N, 4] xyxy)
            and "labels" (Int64Tensor[N], 1-indexed with 0 reserved for
            background).
        categories: Set of category names present in this frame (e.g.
            {"Vehicle", "Pedestrian"}).  Surfaced in per-category filter
            statistics so we can see what the filter selects.
        metadata: Dict with provenance info (frame_id, global_idx, etc.).
    """

    __slots__ = ("image", "annotations", "categories", "metadata")

    def __init__(
        self,
        image: torch.Tensor,
        annotations: Dict[str, torch.Tensor],
        categories: Set[str],
        metadata: Dict[str, Any],
    ):
        self.image = image
        self.annotations = annotations
        self.categories = categories
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain-dict format (e.g. for buffer storage)."""
        return {
            "image": self.image,
            "annotations": {
                "boxes": self.annotations["boxes"],
                "labels": self.annotations["labels"],
            },
            "categories": self.categories,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        shape = tuple(self.image.shape) if isinstance(self.image, torch.Tensor) else "?"
        n_boxes = len(self.annotations["boxes"])
        frame_id = self.metadata.get("frame_id", "?")
        cats = ",".join(sorted(self.categories)) if self.categories else "none"
        return (
            f"StreamItem(image={shape}, boxes={n_boxes}, "
            f"cats=[{cats}], frame={frame_id})"
        )
