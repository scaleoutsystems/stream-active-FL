"""
Training buffer for buffer-based streaming learning.

Accumulates accepted stream items until full, then provides them as a
batch for a training step.  After training the buffer is cleared and
accumulation resumes.  No replay, no sampling -- just sequential
accumulation and batch retrieval.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ..core.items import StreamItem


class TrainingBuffer:
    """
    Fixed-capacity buffer that accumulates StreamItems for batch training.

    Usage:
        buffer = TrainingBuffer(capacity=10)

        for item in stream:
            if policy.accept(item):
                buffer.add(item)

            if buffer.is_full():
                images, targets = buffer.get_batch()
                # ... train on batch ...
                buffer.clear()

    Args:
        capacity: Maximum number of items before the buffer is considered
            full and a training step should be triggered.
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self._items: List[StreamItem] = []
        self.total_flushes = 0
        self.total_items_added = 0

    def add(self, item: StreamItem) -> None:
        """Append a StreamItem to the buffer."""
        self._items.append(item)
        self.total_items_added += 1

    def is_full(self) -> bool:
        """True when the buffer has reached capacity."""
        return len(self._items) >= self.capacity

    def get_batch(self) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Collate all buffered items into a detection training batch.

        Returns:
            (images, targets) where:
            - images: List of tensors, each (3, H, W)
            - targets: List of dicts, each with "boxes" and "labels"
        """
        images = [item.image for item in self._items]
        targets = [
            {
                "boxes": item.annotations["boxes"],
                "labels": item.annotations["labels"],
            }
            for item in self._items
        ]
        return images, targets

    def clear(self) -> None:
        """Clear the buffer after a training step."""
        self._items.clear()
        self.total_flushes += 1

    def __len__(self) -> int:
        return len(self._items)

    def get_stats(self) -> Dict[str, int]:
        return {
            "current_size": len(self._items),
            "capacity": self.capacity,
            "total_flushes": self.total_flushes,
            "total_items_added": self.total_items_added,
        }

    def __repr__(self) -> str:
        return f"TrainingBuffer(size={len(self)}/{self.capacity}, flushes={self.total_flushes})"
