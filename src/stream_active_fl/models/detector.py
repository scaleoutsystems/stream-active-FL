"""
Object detector using torchvision's FCOS.

Provides an FCOS-based detector with a ResNet50-FPN backbone, suitable for
both offline and streaming learning. The backbone can be frozen while the
FPN and detection head remain trainable, giving a middle ground between the
few-parameter classification head and full end-to-end training.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fcos_resnet50_fpn


class Detector(nn.Module):
    """
    Object detector for streaming learning, using FCOS with ResNet50-FPN.

    Uses torchvision's FCOS implementation with configurable backbone freezing.
    The backbone gets pretrained ImageNet weights; the FPN and detection head
    are randomly initialized and trainable.

    In training mode: forward() returns a dict of losses.
    In eval mode: forward() returns a list of prediction dicts.

    Args:
        num_classes: Number of object classes including background.
            For 3 categories (person, car, traffic_light): num_classes=4.
        trainable_backbone_layers: Number of ResNet stages to make trainable
            (0-5, counted from the output end). 0 = fully frozen backbone.
        image_min_size: Minimum side length for the model's internal resizing.
            Set to the shorter image dimension to avoid unnecessary rescaling.
        image_max_size: Maximum side length for the model's internal resizing.
            Set to the longer image dimension to avoid unnecessary rescaling.
        pretrained_backbone: Use ImageNet-pretrained ResNet50 backbone.
    """

    def __init__(
        self,
        num_classes: int = 4,
        trainable_backbone_layers: int = 0,
        image_min_size: int = 360,
        image_max_size: int = 640,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self._trainable_backbone_layers = trainable_backbone_layers
        self.image_min_size = image_min_size
        self.image_max_size = image_max_size

        weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None

        self.model = fcos_resnet50_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=image_min_size,
            max_size=image_max_size,
        )

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        Forward pass.

        Args:
            images: List of image tensors, each (3, H, W) in [0, 1] range.
            targets: Optional list of target dicts, each with:
                - "boxes": FloatTensor[N, 4] in (x1, y1, x2, y2) format
                - "labels": Int64Tensor[N]
                Required in training mode.

        Returns:
            Training: Dict of losses {"classification", "bbox_regression", "bbox_ctrness"}.
            Eval: List of prediction dicts with "boxes", "scores", "labels".
        """
        return self.model(images, targets)

    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Return count of total parameters."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        trainable = self.get_trainable_params()
        total = self.get_total_params()
        return (
            f"Detector(\n"
            f"  backbone=resnet50_fpn,\n"
            f"  num_classes={self.num_classes},\n"
            f"  trainable_backbone_layers={self._trainable_backbone_layers},\n"
            f"  image_min_size={self.image_min_size}, image_max_size={self.image_max_size},\n"
            f"  trainable_params={trainable:,} / {total:,} "
            f"({100 * trainable / total:.1f}%)\n"
            f")"
        )
