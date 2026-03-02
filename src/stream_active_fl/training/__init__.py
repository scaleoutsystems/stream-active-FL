"""
Training loops for the two-phase streaming detection pipeline.

- Bootstrap: multi-epoch supervised training on initial frames
- Streaming: single-pass buffer-based training with filtering
- Federated: FedAvg aggregation (kept for future use)
"""

from .federated import fedavg
from .streaming import (
    StreamingTrainResult,
    bootstrap_train,
    collect_embeddings,
    train_on_stream,
)

__all__ = [
    "StreamingTrainResult",
    "bootstrap_train",
    "collect_embeddings",
    "fedavg",
    "train_on_stream",
]
