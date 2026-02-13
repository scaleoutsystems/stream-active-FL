"""Core data structures for stream learning."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class StreamItem:
    """
    A single item in the stream (frame with metadata).

    For classification tasks, only image/target/teacher_score are used.
    For detection tasks, annotations contains per-frame bounding boxes
    and class labels in torchvision format.
    """

    def __init__(
        self,
        image: torch.Tensor,
        target: float,
        teacher_score: float,
        metadata: Dict[str, Any],
        annotations: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.image = image
        self.target = target
        self.teacher_score = teacher_score
        self.metadata = metadata
        self.annotations = annotations  # {"boxes": Tensor[N,4] xyxy, "labels": Tensor[N]}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for replay buffer storage)."""
        d = {
            "image": self.image,
            "target": torch.tensor(self.target, dtype=torch.float32),
            "teacher_score": self.teacher_score,
            "metadata": self.metadata,
        }
        if self.annotations is not None:
            d["annotations"] = {
                "boxes": self.annotations["boxes"],
                "labels": self.annotations["labels"],
            }
        return d