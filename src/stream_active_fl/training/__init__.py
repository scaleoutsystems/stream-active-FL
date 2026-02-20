"""
Training abstractions and loops.

Reusable training logic for different learning paradigms:
- Streaming: Online training from temporal data streams
- Federated: Distributed training with periodic aggregation (future)
"""

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
    "perform_classification_update",
    "perform_detection_update",
    "train_on_classification_stream",
    "train_on_detection_stream",
]
