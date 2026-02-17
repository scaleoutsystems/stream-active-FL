"""
Model definitions.

    Classifier  Binary classification (ResNet backbone + linear head)
    Detector    Object detection (FCOS with ResNet50-FPN backbone)
"""

from .classifier import Classifier
from .detector import Detector

__all__ = ["Classifier", "Detector"]
