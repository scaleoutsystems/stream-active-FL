"""Tests for streaming local training modes."""

from __future__ import annotations

import torch
import torch.nn as nn

from stream_active_fl.core.items import StreamItem
from stream_active_fl.memory import TrainingBuffer
from stream_active_fl.policies import NoFilterPolicy
from stream_active_fl.training import train_on_stream


class DummyDetector(nn.Module):
    """Minimal detector-like module returning a scalar loss dict."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets):  # noqa: D401
        # Loss depends on parameter so backward() and optimizer.step() are valid.
        s = torch.stack([img.mean() for img in images]).sum()
        return {"loss_total": self.w * s}


def _make_stream_item(i: int) -> StreamItem:
    return StreamItem(
        image=torch.rand(3, 16, 16),
        annotations={
            "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
        categories={"Vehicle"},
        metadata={"frame_id": f"f{i}", "global_idx": i},
    )


def test_full_batch_train_steps_per_buffer():
    model = DummyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    buffer = TrainingBuffer(capacity=4)
    stream = [_make_stream_item(i) for i in range(4)]

    result = train_on_stream(
        model=model,
        stream=stream,
        optimizer=optimizer,
        filter_policy=NoFilterPolicy(),
        training_buffer=buffer,
        device=torch.device("cpu"),
        train_steps_per_buffer=3,
        buffer_training_mode="full_batch",
        progress_bar=False,
    )

    assert result.items_processed == 4
    assert result.items_accepted == 4
    assert result.optimizer_steps == 3
    assert result.buffer_flushes == 1


def test_minibatch_local_epochs_with_reshuffle():
    model = DummyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    buffer = TrainingBuffer(capacity=4)
    stream = [_make_stream_item(i) for i in range(4)]

    result = train_on_stream(
        model=model,
        stream=stream,
        optimizer=optimizer,
        filter_policy=NoFilterPolicy(),
        training_buffer=buffer,
        device=torch.device("cpu"),
        buffer_training_mode="mini_batch",
        local_epochs_per_buffer=2,
        mini_batch_size=2,
        shuffle_buffer_each_epoch=True,
        progress_bar=False,
    )

    # 4 items, mini_batch_size=2 => 2 mini-batches/epoch, 2 epochs => 4 steps.
    assert result.items_processed == 4
    assert result.items_accepted == 4
    assert result.optimizer_steps == 4
    assert result.buffer_flushes == 1
