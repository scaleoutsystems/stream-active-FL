"""
Training buffer for buffer-based streaming learning.

Accumulates accepted stream items until full, then provides them as a
batch for a training step.  After training the buffer is cleared and
accumulation resumes.  No replay, no sampling -- just sequential
accumulation and batch retrieval.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

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
        return self._collate_items(self._items)

    def get_minibatches(
        self,
        mini_batch_size: int,
        *,
        shuffle: bool = False,
    ) -> List[Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]]:
        """
        Split current buffer content into mini-batches.

        Args:
            mini_batch_size: Number of items per mini-batch (> 0).
            shuffle: If True, randomize item order before splitting.

        Returns:
            List of mini-batches, each in the same format as get_batch().
        """
        if mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be > 0")
        if len(self._items) == 0:
            return []

        items = list(self._items)
        if shuffle:
            random.shuffle(items)

        batches: List[Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]] = []
        for start in range(0, len(items), mini_batch_size):
            batch_items = items[start : start + mini_batch_size]
            batches.append(self._collate_items(batch_items))
        return batches

    @staticmethod
    def _collate_items(
        items: List[StreamItem],
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        images = [item.image for item in items]
        targets = [
            {
                "boxes": item.annotations["boxes"],
                "labels": item.annotations["labels"],
            }
            for item in items
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
