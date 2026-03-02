"""
Object detector using torchvision's FCOS with embedding extraction.

Provides an FCOS-based detector with a ResNet50-FPN backbone, suitable for
both offline and streaming learning.  Includes a get_embedding() method that
extracts global-average-pooled backbone features for distribution-based
filtering policies.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn


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
            For 10 ZOD categories: num_classes=11.
        trainable_backbone_layers: Number of ResNet stages to make trainable
            (0-5, counted from the output end). 0 = fully frozen backbone.
        image_min_size: Minimum side length for the model's internal resizing.
        image_max_size: Maximum side length for the model's internal resizing.
        pretrained_backbone: Use ImageNet-pretrained ResNet50 backbone.
        pretrained_detector: Initialize from COCO FCOS weights where shapes match
            (backbone/FPN/regression branches). Classification head is reinitialized
            for custom num_classes.
    """

    def __init__(
        self,
        num_classes: int = 11,
        trainable_backbone_layers: int = 0,
        image_min_size: int = 720,
        image_max_size: int = 1280,
        pretrained_backbone: bool = True,
        pretrained_detector: bool = False,
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

        if pretrained_detector:
            self._load_partial_coco_weights()

        # Hook state for embedding extraction
        self._embedding_hook = None
        self._embedding_output: Optional[torch.Tensor] = None

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

    def _load_partial_coco_weights(self) -> None:
        """Load COCO FCOS weights for all shape-compatible parameters."""
        coco_model = fcos_resnet50_fpn(
            weights=FCOS_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone=None,
        )
        source_state = coco_model.state_dict()
        target_state = self.model.state_dict()

        compatible_state = {
            k: v
            for k, v in source_state.items()
            if k in target_state and target_state[k].shape == v.shape
        }
        self.model.load_state_dict(compatible_state, strict=False)

    @torch.no_grad()
    def get_embedding(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract backbone embeddings via global average pooling.

        Hooks into the backbone's body (ResNet without the final FC) to
        capture the feature map from the last residual stage, then applies
        adaptive average pooling to produce a fixed-size vector per image.

        Args:
            images: List of image tensors, each (3, H, W) in [0, 1] range.

        Returns:
            Tensor of shape (B, 2048) -- one embedding per image.
        """
        was_training = self.training
        self.eval()

        captured = {}

        def hook_fn(module, input, output):
            # output is an OrderedDict from the ResNet body; grab the last layer
            if isinstance(output, dict):
                last_key = list(output.keys())[-1]
                captured["features"] = output[last_key]
            else:
                captured["features"] = output

        backbone_body = self.model.backbone.body
        handle = backbone_body.register_forward_hook(hook_fn)

        try:
            # Run forward (eval mode returns predictions, but we only need the hook)
            self.model(images)
            features = captured["features"]  # (B, C, H, W)
            embeddings = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            embeddings = embeddings.flatten(1)  # (B, 2048)
        finally:
            handle.remove()
            if was_training:
                self.train()

        return embeddings

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
