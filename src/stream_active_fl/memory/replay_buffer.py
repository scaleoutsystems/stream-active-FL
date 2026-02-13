"""
Replay buffer for experience replay in streaming learning.

Maintains a bounded memory buffer of past training items for rehearsal-style updates.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Literal, Optional

import torch


class ReplayBuffer:
    """
    Bounded replay buffer with configurable admission policies.

    Stores stream items (or compact representations) for experience replay.
    When the buffer is full, new items are admitted according to the policy:
    - FIFO: Remove oldest item
    - random: Remove random item
    - reservoir: Reservoir sampling (uniform over stream)

    Args:
        capacity: Maximum number of items to store.
        admission_policy: How to admit items when buffer is full.
        device: Device to store tensors on (for memory efficiency).

    Usage:
        buffer = ReplayBuffer(capacity=1000, admission_policy="fifo")
        buffer.add(stream_item.to_dict())
        
        # Sample for replay during training
        replay_batch = buffer.sample(batch_size=32)
    """

    def __init__(
        self,
        capacity: int,
        admission_policy: Literal["fifo", "random", "reservoir"] = "fifo",
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.admission_policy = admission_policy
        self.device = device

        # Storage
        if admission_policy == "fifo":
            self.buffer = deque(maxlen=capacity)
        else:
            self.buffer: List[Dict[str, Any]] = []

        # Reservoir sampling state
        self.total_seen = 0  # Total items seen (for reservoir sampling)

    def add(self, item: Dict[str, Any]) -> bool:
        """
        Add an item to the replay buffer.

        Args:
            item: Dictionary with keys "image", "target", "teacher_score", "metadata",
                  and optionally "annotations" (for detection tasks).
                  Images should be tensors.

        Returns:
            True if item was added, False if rejected.
        """
        self.total_seen += 1

        # Move tensors to buffer device (for memory efficiency)
        item_copy = {
            "image": item["image"].to(self.device),
            "target": item["target"].to(self.device),
            "teacher_score": item["teacher_score"],
            "metadata": item["metadata"].copy() if isinstance(item["metadata"], dict) else item["metadata"],
        }

        # Store detection annotations if present
        if "annotations" in item and item["annotations"] is not None:
            item_copy["annotations"] = {
                "boxes": item["annotations"]["boxes"].to(self.device),
                "labels": item["annotations"]["labels"].to(self.device),
            }

        if self.admission_policy == "fifo":
            self.buffer.append(item_copy)
            return True

        elif self.admission_policy == "random":
            if len(self.buffer) < self.capacity:
                self.buffer.append(item_copy)
                return True
            else:
                # Replace random item
                idx = random.randint(0, self.capacity - 1)
                self.buffer[idx] = item_copy
                return True

        elif self.admission_policy == "reservoir":
            # Reservoir sampling algorithm
            if len(self.buffer) < self.capacity:
                self.buffer.append(item_copy)
                return True
            else:
                # Probability of admission: capacity / total_seen
                # Equivalent to: choose random idx in [0, total_seen), accept if idx < capacity
                idx = random.randint(0, self.total_seen - 1)
                if idx < self.capacity:
                    self.buffer[idx] = item_copy
                    return True
                return False

        else:
            raise ValueError(f"Unknown admission policy: {self.admission_policy}")

    def sample(self, batch_size: int, device: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Sample a batch from the replay buffer.

        Args:
            batch_size: Number of items to sample.
            device: Device to move sampled batch to (defaults to buffer device).

        Returns:
            For classification: Batched dict with "image", "target", "teacher_score" tensors.
            For detection: Dict with "images" (list of tensors) and "targets" (list of dicts).
            Returns None if buffer is empty.
        """
        if len(self.buffer) == 0:
            return None

        if batch_size > len(self.buffer):
            # Sample all items if requested batch is larger
            batch_size = len(self.buffer)

        # Sample without replacement
        if self.admission_policy == "fifo":
            # Convert deque to list for sampling
            items = random.sample(list(self.buffer), batch_size)
        else:
            items = random.sample(self.buffer, batch_size)

        # Stack into batched tensors
        target_device = device if device is not None else self.device

        # Detection mode: items have variable-length annotations, return lists
        has_annotations = len(items) > 0 and "annotations" in items[0]
        if has_annotations:
            return {
                "images": [item["image"].to(target_device) for item in items],
                "targets": [
                    {
                        "boxes": item["annotations"]["boxes"].to(target_device),
                        "labels": item["annotations"]["labels"].to(target_device),
                    }
                    for item in items
                ],
            }

        # Classification mode: stack into batched tensors (original behavior)
        batch = {
            "image": torch.stack([item["image"] for item in items]).to(target_device),
            "target": torch.stack([item["target"] for item in items]).to(target_device),
            "teacher_score": torch.tensor(
                [item["teacher_score"] for item in items],
                dtype=torch.float32,
                device=target_device,
            ),
        }

        return batch

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) >= self.capacity

    def clear(self) -> None:
        """Clear the buffer."""
        if self.admission_policy == "fifo":
            self.buffer.clear()
        else:
            self.buffer = []
        self.total_seen = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics, including class distribution."""
        n = len(self.buffer)

        # Count positives and negatives in buffer
        n_positive = 0
        if n > 0:
            for item in self.buffer:
                target = item["target"]
                if isinstance(target, torch.Tensor):
                    if target.item() == 1.0:
                        n_positive += 1
                elif target == 1.0:
                    n_positive += 1
        n_negative = n - n_positive

        return {
            "size": n,
            "capacity": self.capacity,
            "utilization": n / self.capacity if self.capacity > 0 else 0.0,
            "total_seen": self.total_seen,
            "admission_policy": self.admission_policy,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "positive_ratio": n_positive / n if n > 0 else 0.0,
        }
