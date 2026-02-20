"""
Training abstractions and loops.

Reusable training logic for different learning paradigms:
- Streaming: Online training from temporal data streams
- Federated: Server-side aggregation (FedAvg) for simulated FL
"""

from .federated import fedavg
from .streaming import (
    RunningPosWeight,
    StreamingTrainResult,
    perform_classification_update,
    perform_detection_update,
    train_on_classification_stream,
    train_on_detection_stream,
)

__all__ = [
    "RunningPosWeight",
    "StreamingTrainResult",
    "fedavg",
    "perform_classification_update",
    "perform_detection_update",
    "train_on_classification_stream",
    "train_on_detection_stream",
]
