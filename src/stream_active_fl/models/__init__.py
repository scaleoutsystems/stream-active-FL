"""
Model definitions.

    Detector    Object detection (FCOS with ResNet50-FPN backbone + embedding extraction)
"""

from .detector import Detector

__all__ = ["Detector"]
