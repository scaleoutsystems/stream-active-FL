"""
Core data structures for stream learning.

Defines the StreamItem class, the fundamental unit of data flowing through
the streaming pipeline. Each StreamItem represents a single camera frame
with its associated labels and metadata.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class StreamItem:
    """
    A single camera frame in the data stream.

    Carries both classification and detection information so the same
    StreamItem can be consumed by either pipeline:

    - **Classification**: Uses ``image``, ``target`` (binary 0/1), and
      ``teacher_score`` (max pseudo-label confidence for the target category).
    - **Detection**: Additionally uses ``annotations``, a dict with
      ``"boxes"`` (FloatTensor[N, 4] in xyxy format) and ``"labels"``
      (Int64Tensor[N], 1-indexed with 0 reserved for background).

    Attributes:
        image: Image tensor of shape (C, H, W).
        target: Binary label (1.0 = at least one object present, 0.0 = empty).
        teacher_score: Maximum pseudo-label confidence score for the frame
            (0.0 when no detections exist).
        metadata: Dict with provenance info (seq_id, frame_idx, etc.).
        annotations: Detection targets, or None for classification-only items.
    """

    __slots__ = ("image", "target", "teacher_score", "metadata", "annotations")

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
        self.annotations = annotations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for replay buffer storage."""
        d: Dict[str, Any] = {
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

    def __repr__(self) -> str:
        shape = tuple(self.image.shape) if isinstance(self.image, torch.Tensor) else "?"
        n_boxes = len(self.annotations["boxes"]) if self.annotations is not None else None
        parts = [
            f"image={shape}",
            f"target={self.target}",
            f"teacher_score={self.teacher_score:.2f}",
        ]
        if n_boxes is not None:
            parts.append(f"boxes={n_boxes}")
        seq_id = self.metadata.get("seq_id", "?")
        frame_idx = self.metadata.get("frame_idx", "?")
        parts.append(f"seq={seq_id}[{frame_idx}]")
        return f"StreamItem({', '.join(parts)})"