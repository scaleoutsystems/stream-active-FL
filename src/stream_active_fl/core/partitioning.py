"""
Client partitioning for federated learning simulation.

Supports two strategies:
  - contiguous: equal-sized consecutive slices (original behavior).
  - domain_aligned: each client receives one or more named domain
    blocks from the manifest, giving every client a coherent deployment
    environment (e.g. city, suburban, highway, rural).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Sequence, Tuple


def partition_frames(
    num_frames: int,
    num_clients: int,
    strategy: Literal["uniform", "contiguous"] = "contiguous",
) -> Dict[int, Tuple[int, int]]:
    """
    Partition frame indices into equal-sized per-client ranges.

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


def partition_frames_by_domain(
    block_order: Sequence[str],
    block_sizes: Dict[str, int],
    client_groups: Sequence[Sequence[str]],
) -> Dict[int, Tuple[int, int]]:
    """Partition the stream so each client receives specific domain blocks.

    Args:
        block_order: Ordered list of block names as they appear in the
            stream (from the manifest's ordering.block_order).
        block_sizes: Mapping block_name -> number of frames (from
            ordering.block_sizes).
        client_groups: One entry per client.  Each entry is a list of
            block names assigned to that client.  The blocks within each
            group must be adjacent in block_order.

    Returns:
        Dict mapping client_id -> (start_idx, end_idx) half-open range
        (same contract as partition_frames).

    Raises:
        ValueError: If a block name is unknown, blocks within a group are
            non-adjacent, or any block is assigned to more than one client.
    """
    block_offset: Dict[str, int] = {}
    offset = 0
    for name in block_order:
        block_offset[name] = offset
        offset += block_sizes[name]

    assigned: set[str] = set()
    partitions: Dict[int, Tuple[int, int]] = {}

    for cid, group in enumerate(client_groups):
        if not group:
            raise ValueError(f"Client {cid} has an empty block group")

        for bname in group:
            if bname not in block_sizes:
                raise ValueError(
                    f"Block '{bname}' (client {cid}) not in manifest "
                    f"block_sizes {list(block_sizes)}"
                )
            if bname in assigned:
                raise ValueError(f"Block '{bname}' assigned to multiple clients")
            assigned.add(bname)

        indices = [block_order.index(b) for b in group]
        if sorted(indices) != list(range(min(indices), max(indices) + 1)):
            raise ValueError(
                f"Blocks {group} for client {cid} are not adjacent in "
                f"block_order {list(block_order)}"
            )

        first_block = block_order[min(indices)]
        last_block = block_order[max(indices)]
        start = block_offset[first_block]
        end = block_offset[last_block] + block_sizes[last_block]
        partitions[cid] = (start, end)

    return partitions
