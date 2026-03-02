"""
Client partitioning for federated learning simulation.

Partitions the chronologically-sorted frame stream across simulated clients
by contiguous frame-ID ranges, so each client sees a distinct temporal slice.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple


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
            chronological stream.  "uniform" is identical for this case
            (both produce non-overlapping contiguous ranges).

    Returns:
        Dict mapping client_id -> (start_idx, end_idx) half-open range.
    """
    base = num_frames // num_clients
    remainder = num_frames % num_clients

    partitions: Dict[int, Tuple[int, int]] = {}
    offset = 0
    for client_id in range(num_clients):
        size = base + (1 if client_id < remainder else 0)
        partitions[client_id] = (offset, offset + size)
        offset += size

    return partitions
