"""Tests for stream_active_fl.core.partitioning."""

from __future__ import annotations

import pytest

from stream_active_fl.core.partitioning import (
    partition_frames,
    partition_frames_by_domain,
)


# ---------------------------------------------------------------------------
# partition_frames
# ---------------------------------------------------------------------------


def test_even_split():
    parts = partition_frames(num_frames=100, num_clients=4)
    assert len(parts) == 4
    for cid in range(4):
        s, e = parts[cid]
        assert e - s == 25
    assert parts[0] == (0, 25)
    assert parts[3] == (75, 100)


def test_uneven_split_distributes_remainder():
    parts = partition_frames(num_frames=10, num_clients=3)
    sizes = [e - s for s, e in parts.values()]
    assert sum(sizes) == 10
    assert sizes[0] == 4  # first client gets the extra frame
    assert sizes[1] == 3
    assert sizes[2] == 3


def test_single_client_gets_all():
    parts = partition_frames(num_frames=50, num_clients=1)
    assert parts[0] == (0, 50)


def test_more_clients_than_frames():
    parts = partition_frames(num_frames=2, num_clients=5)
    sizes = [e - s for s, e in parts.values()]
    assert sum(sizes) == 2
    non_empty = [s for s in sizes if s > 0]
    assert len(non_empty) == 2


def test_zero_frames():
    parts = partition_frames(num_frames=0, num_clients=3)
    for cid in range(3):
        s, e = parts[cid]
        assert s == e == 0


def test_invalid_num_clients():
    with pytest.raises(ValueError, match="num_clients must be > 0"):
        partition_frames(num_frames=10, num_clients=0)


def test_contiguous_no_gaps_no_overlaps():
    parts = partition_frames(num_frames=37, num_clients=5)
    ranges = [parts[i] for i in range(5)]
    for i in range(len(ranges) - 1):
        assert ranges[i][1] == ranges[i + 1][0]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 37


def test_uniform_strategy_warns_and_partitions_contiguously():
    with pytest.warns(UserWarning, match="behaves like 'contiguous'"):
        parts = partition_frames(num_frames=12, num_clients=3, strategy="uniform")
    assert parts == {0: (0, 4), 1: (4, 8), 2: (8, 12)}


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="Unknown partition strategy"):
        partition_frames(num_frames=10, num_clients=2, strategy="random")  # type: ignore[arg-type]


def test_negative_num_frames_rejected():
    with pytest.raises(ValueError, match="num_frames must be >= 0"):
        partition_frames(num_frames=-1, num_clients=2)


# ---------------------------------------------------------------------------
# partition_frames_by_domain
# ---------------------------------------------------------------------------


def _abc_blocks(a: int = 30, b: int = 50, c: int = 20) -> dict[str, int]:
    return {"A": a, "B": b, "C": c}


def test_by_domain_single_block_per_client():
    parts = partition_frames_by_domain(
        block_order=["A", "B", "C"],
        block_sizes=_abc_blocks(),
        client_groups=[["A"], ["B"], ["C"]],
    )
    assert parts == {0: (0, 30), 1: (30, 80), 2: (80, 100)}


def test_by_domain_multi_block_group_concatenated():
    parts = partition_frames_by_domain(
        block_order=["A", "B", "C"],
        block_sizes=_abc_blocks(),
        client_groups=[["A", "B"], ["C"]],
    )
    assert parts == {0: (0, 80), 1: (80, 100)}


def test_by_domain_group_order_independent_of_input_order():
    """Adjacency is determined by block_order, not the order inside a group."""
    parts = partition_frames_by_domain(
        block_order=["A", "B", "C"],
        block_sizes=_abc_blocks(),
        client_groups=[["B", "A"], ["C"]],  # A and B are still adjacent in stream
    )
    assert parts == {0: (0, 80), 1: (80, 100)}


def test_by_domain_non_adjacent_blocks_rejected():
    with pytest.raises(ValueError, match="not adjacent"):
        partition_frames_by_domain(
            block_order=["A", "B", "C"],
            block_sizes=_abc_blocks(),
            client_groups=[["A", "C"]],
        )


def test_by_domain_unknown_block_rejected():
    with pytest.raises(ValueError, match="not in manifest"):
        partition_frames_by_domain(
            block_order=["A", "B"],
            block_sizes={"A": 10, "B": 10},
            client_groups=[["X"]],
        )


def test_by_domain_block_assigned_to_two_clients_rejected():
    with pytest.raises(ValueError, match="multiple clients"):
        partition_frames_by_domain(
            block_order=["A", "B"],
            block_sizes={"A": 10, "B": 10},
            client_groups=[["A"], ["A", "B"]],
        )


def test_by_domain_empty_group_rejected():
    with pytest.raises(ValueError, match="empty block group"):
        partition_frames_by_domain(
            block_order=["A"],
            block_sizes={"A": 5},
            client_groups=[[]],
        )


def test_by_domain_unassigned_blocks_left_uncovered():
    """Blocks not listed in any client_group are simply not assigned.

    The federated experiment relies on this to drop trailing blocks (e.g.
    novel domains the curated manifest reserves for evaluation only).
    """
    parts = partition_frames_by_domain(
        block_order=["A", "B", "C"],
        block_sizes=_abc_blocks(),
        client_groups=[["A"], ["B"]],
    )
    assert parts == {0: (0, 30), 1: (30, 80)}
    covered = sum(e - s for s, e in parts.values())
    assert covered == 30 + 50  # block C (20 frames) is uncovered
