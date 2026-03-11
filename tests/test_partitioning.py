"""Tests for stream_active_fl.core.partitioning."""

from __future__ import annotations

import pytest

from stream_active_fl.core.partitioning import partition_frames


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
