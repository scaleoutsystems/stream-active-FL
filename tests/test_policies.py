"""Tests for filter policies (NoFilterPolicy, RandomPolicy)."""

from __future__ import annotations

import torch
import torch.nn as nn

from stream_active_fl.core.items import StreamItem
from stream_active_fl.policies.filtering import NoFilterPolicy, RandomPolicy


def _make_item(categories: set[str] | None = None) -> StreamItem:
    return StreamItem(
        image=torch.rand(3, 32, 32),
        annotations={
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),
        },
        categories=categories or {"Vehicle"},
        metadata={"frame_id": "test"},
    )


class _DummyModel(nn.Module):
    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# NoFilterPolicy
# ---------------------------------------------------------------------------


def test_no_filter_always_accepts():
    policy = NoFilterPolicy()
    model = _DummyModel()
    device = torch.device("cpu")

    for _ in range(10):
        action, meta = policy.select_action(_make_item(), model, device)
        assert action == "accept"

    assert policy.count == 10
    assert policy.requires_model_forward() is False


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------


def test_random_policy_respects_fraction():
    policy = RandomPolicy(accept_fraction=0.5)
    model = _DummyModel()
    device = torch.device("cpu")

    n = 1000
    accepts = 0
    for _ in range(n):
        action, meta = policy.select_action(_make_item(), model, device)
        assert action in ("accept", "reject")
        assert "random_score" in meta
        if action == "accept":
            accepts += 1

    rate = accepts / n
    assert 0.35 < rate < 0.65, f"Expected ~0.5, got {rate}"


def test_random_policy_zero_fraction_rejects_all():
    policy = RandomPolicy(accept_fraction=0.0)
    model = _DummyModel()
    device = torch.device("cpu")

    for _ in range(50):
        action, _ = policy.select_action(_make_item(), model, device)
        assert action == "reject"


def test_random_policy_one_fraction_accepts_all():
    policy = RandomPolicy(accept_fraction=1.0)
    model = _DummyModel()
    device = torch.device("cpu")

    for _ in range(50):
        action, _ = policy.select_action(_make_item(), model, device)
        assert action == "accept"


def test_random_policy_requires_no_forward():
    assert RandomPolicy().requires_model_forward() is False


def test_random_policy_stats():
    policy = RandomPolicy(accept_fraction=1.0)
    model = _DummyModel()
    device = torch.device("cpu")

    for _ in range(5):
        policy.select_action(_make_item(), model, device)

    stats = policy.get_stats()
    assert stats["count_accept"] == 5
    assert stats["count_reject"] == 0
    assert stats["accept_rate"] == 1.0


# ---------------------------------------------------------------------------
# SelectionTracker (via policy)
# ---------------------------------------------------------------------------


def test_selection_tracker_per_category():
    policy = NoFilterPolicy()
    model = _DummyModel()
    device = torch.device("cpu")

    policy.select_action(_make_item({"Vehicle"}), model, device)
    policy.select_action(_make_item({"Pedestrian"}), model, device)
    policy.select_action(_make_item({"Vehicle", "Pedestrian"}), model, device)

    stats = policy.get_selection_stats()
    assert stats["accept_count"] == 3
    assert stats["accept_by_category"]["Vehicle"] == 2
    assert stats["accept_by_category"]["Pedestrian"] == 2


def test_selection_tracker_reset():
    policy = NoFilterPolicy()
    model = _DummyModel()
    device = torch.device("cpu")

    policy.select_action(_make_item(), model, device)
    policy.reset_selection_stats()

    stats = policy.get_selection_stats()
    assert stats["accept_count"] == 0
    assert stats["reject_count"] == 0
