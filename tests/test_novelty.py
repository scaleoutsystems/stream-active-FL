"""Tests for stream_active_fl.evaluation.novelty.NoveltyTracker."""

from __future__ import annotations

from stream_active_fl.evaluation.novelty import NoveltyTracker


def test_first_category_is_novel():
    tracker = NoveltyTracker()
    tracker.observe({"Vehicle"}, "accept")
    assert tracker.last_was_novel is True
    assert tracker.novel_total == 1
    assert tracker.novel_accepted == 1


def test_seen_category_is_redundant():
    tracker = NoveltyTracker()
    tracker.observe({"Vehicle"}, "accept")
    tracker.observe({"Vehicle"}, "reject")

    assert tracker.last_was_novel is False
    assert tracker.redundant_total == 1
    assert tracker.redundant_rejected == 1


def test_empty_frame():
    tracker = NoveltyTracker()
    tracker.observe(set(), "accept")

    assert tracker.empty_total == 1
    assert tracker.empty_accepted == 1
    assert tracker.last_was_novel is False


def test_mixed_novel_and_known():
    tracker = NoveltyTracker()
    tracker.observe({"Vehicle"}, "accept")
    tracker.observe({"Vehicle", "Pedestrian"}, "accept")

    assert tracker.last_was_novel is True
    assert tracker.novel_total == 2
    assert tracker.seen_categories == {"Vehicle", "Pedestrian"}


def test_categories_tracked_regardless_of_action():
    tracker = NoveltyTracker()
    tracker.observe({"Vehicle"}, "reject")
    assert "Vehicle" in tracker.seen_categories
    tracker.observe({"Vehicle"}, "accept")
    assert tracker.redundant_total == 1


def test_get_stats_keys():
    tracker = NoveltyTracker()
    tracker.observe({"A"}, "accept")
    tracker.observe({"B"}, "reject")
    tracker.observe(set(), "reject")

    stats = tracker.get_stats()
    assert stats["categories_seen"] == 2
    assert "A" in stats["categories_list"]
    assert stats["novel_total"] == 2
    assert stats["novel_accepted"] == 1
    assert stats["novel_rejected"] == 1
    assert stats["empty_total"] == 1
    assert stats["empty_rejected"] == 1
    assert 0.0 <= stats["novel_accept_rate"] <= 1.0
    assert 0.0 <= stats["redundant_reject_rate"] <= 1.0
