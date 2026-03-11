"""
Client partitioning for federated learning simulation.

Partitions the chronologically-sorted frame stream across simulated clients
by contiguous frame-ID ranges, so each client sees a distinct temporal slice.
"""

from __future__ import annotations

import warnings
from typing import Dict, Literal, Tuple


def partition_frames(
    num_frames: int,
    num_clients: int,
    strategy: Literal["uniform", "contiguous"] = "contiguous",
) -> Dict[int, Tuple[int, int]]:
    """
    Partition frame indices into per-client ranges.

    Args:
        num_frames: Total number of frames in the stream.
        num_clients: Number of simulated clients.
        strategy: "contiguous" gives each client a contiguous slice of the
            chronological stream. "uniform" is currently an alias for
            "contiguous" and is kept only for backward compatibility.

    Returns:
        Dict mapping client_id -> (start_idx, end_idx) half-open range.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if num_frames < 0:
        raise ValueError("num_frames must be >= 0")
    if strategy not in ("contiguous", "uniform"):
        raise ValueError(f"Unknown partition strategy: {strategy}")
    if strategy == "uniform":
        warnings.warn(
            "partition strategy 'uniform' currently behaves like 'contiguous'.",
            stacklevel=2,
        )

    base = num_frames // num_clients
    remainder = num_frames % num_clients

    partitions: Dict[int, Tuple[int, int]] = {}
    offset = 0
    for client_id in range(num_clients):
        size = base + (1 if client_id < remainder else 0)
        partitions[client_id] = (offset, offset + size)
        offset += size

    return partitions
