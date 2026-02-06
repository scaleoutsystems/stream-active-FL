"""Core data structures for stream learning."""

from __future__ import annotations

from typing import Any, Dict

import torch


class StreamItem:
    """A single item in the stream (frame with metadata)."""

    def __init__(
        self,
        image: torch.Tensor,
        target: float,
        teacher_score: float,
        metadata: Dict[str, Any],
    ):
        self.image = image
        self.target = target
        self.teacher_score = teacher_score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for replay buffer storage)."""
        return {
            "image": self.image,
            "target": torch.tensor(self.target, dtype=torch.float32),
            "teacher_score": self.teacher_score,
            "metadata": self.metadata,
        }
