"""Tests for stream_active_fl.memory.TrainingBuffer."""

from __future__ import annotations

import torch

from stream_active_fl.core.items import StreamItem
from stream_active_fl.memory import TrainingBuffer


def _make_item(n_boxes: int = 2, frame_id: str = "test") -> StreamItem:
    return StreamItem(
        image=torch.rand(3, 64, 64),
        annotations={
            "boxes": torch.rand(n_boxes, 4),
            "labels": torch.ones(n_boxes, dtype=torch.int64),
        },
        categories={"Vehicle"},
        metadata={"frame_id": frame_id},
    )


def test_add_and_len():
    buf = TrainingBuffer(capacity=4)
    assert len(buf) == 0
    buf.add(_make_item())
    assert len(buf) == 1


def test_is_full():
    buf = TrainingBuffer(capacity=2)
    buf.add(_make_item())
    assert not buf.is_full()
    buf.add(_make_item())
    assert buf.is_full()


def test_get_batch_returns_correct_structure():
    buf = TrainingBuffer(capacity=3)
    for i in range(3):
        buf.add(_make_item(n_boxes=i + 1, frame_id=f"f{i}"))

    images, targets = buf.get_batch()
    assert len(images) == 3
    assert len(targets) == 3
    assert images[0].shape[0] == 3  # channels
    assert "boxes" in targets[0]
    assert "labels" in targets[0]
    assert targets[1]["boxes"].shape[0] == 2


def test_clear_resets_buffer():
    buf = TrainingBuffer(capacity=2)
    buf.add(_make_item())
    buf.add(_make_item())
    assert buf.is_full()

    buf.clear()
    assert len(buf) == 0
    assert not buf.is_full()
    assert buf.total_flushes == 1


def test_stats_tracking():
    buf = TrainingBuffer(capacity=2)
    buf.add(_make_item())
    buf.add(_make_item())
    buf.clear()
    buf.add(_make_item())

    stats = buf.get_stats()
    assert stats["total_items_added"] == 3
    assert stats["total_flushes"] == 1
    assert stats["current_size"] == 1
    assert stats["capacity"] == 2
