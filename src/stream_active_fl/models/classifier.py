"""Binary classifier with frozen backbone and trainable head."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torchvision import models


class Classifier(nn.Module):
    """
    Binary classifier using a pretrained backbone with a linear head.

    Args:
        backbone: Which torchvision model to use as feature extractor.
        pretrained: Whether to load pretrained ImageNet weights.
        freeze_backbone: If True, freeze all backbone parameters (train head only).
        dropout: Dropout probability before the final linear layer.

    Forward returns:
        Logits of shape (batch_size,) for binary classification.
        Use BCEWithLogitsLoss for training.
    """

    SUPPORTED_BACKBONES = ("resnet18", "resnet34", "resnet50", "resnet101")

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "resnet50", "resnet101"] = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from: {self.SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone

        # Load pretrained model
        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=weights)
        elif backbone == "resnet101":
            self.backbone = models.resnet101(weights=weights)

        # Get feature dimension and remove the original fc layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Keep BatchNorm in eval mode to use pretrained running statistics
            # instead of batch statistics, ensuring consistent behavior
            self.backbone.eval()

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.feature_dim, 1),
        )

    def train(self, mode: bool = True) -> "Classifier":
        """
        Set the module in training mode.

        Overridden to keep the frozen backbone in eval mode, ensuring BatchNorm
        layers use pretrained running statistics instead of batch statistics.
        """
        super().train(mode)
        if self.freeze_backbone:
            # Keep backbone in eval mode even when training the head
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (batch_size, 3, H, W).

        Returns:
            Logits of shape (batch_size,).
        """
        features = self.backbone(x)  # (batch_size, feature_dim)
        logits = self.head(features)  # (batch_size, 1)
        return logits.squeeze(-1)  # (batch_size,)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False

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
            f"Classifier(\n"
            f"  backbone={self.backbone_name},\n"
            f"  freeze_backbone={self.freeze_backbone},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  trainable_params={trainable:,} / {total:,} "
            f"({100 * trainable / total:.1f}%)\n"
            f")"
        )
