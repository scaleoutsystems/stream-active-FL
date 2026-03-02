"""
Novelty tracking for streaming active learning evaluation.

Tracks which object categories have been seen in the stream so far and
evaluates how well the filter policy captures frames containing novel
(previously unseen) categories.

Key metrics:
- novel_accept_rate: Fraction of novel-category frames that were accepted
- redundant_reject_rate: Fraction of already-seen-category frames that were rejected
"""

from __future__ import annotations

from typing import Any, Dict, Set


class NoveltyTracker:
    """
    Tracks category novelty and filter decisions over the stream.

    A frame is "novel" if it contains at least one category that has not
    appeared in any earlier frame in the stream.  A frame is "redundant"
    if all its categories have been seen before.

    Attributes:
        last_was_novel: Whether the most recently observed frame was novel.
    """

    def __init__(self):
        self.seen_categories: Set[str] = set()
        self.last_was_novel: bool = False

        # Novel frames
        self.novel_total = 0
        self.novel_accepted = 0
        self.novel_rejected = 0

        # Redundant frames (all categories already seen)
        self.redundant_total = 0
        self.redundant_accepted = 0
        self.redundant_rejected = 0

        # Frames with no annotations
        self.empty_total = 0
        self.empty_accepted = 0
        self.empty_rejected = 0

    def observe(self, categories: Set[str], action: str) -> None:
        """
        Record a stream item's categories and the filter's decision.

        Args:
            categories: Set of category names present in the frame.
            action: "accept" or "reject".
        """
        is_accepted = (action == "accept")

        if not categories:
            self.empty_total += 1
            if is_accepted:
                self.empty_accepted += 1
            else:
                self.empty_rejected += 1
            self.last_was_novel = False
            return

        # Check if any category is novel (never seen before)
        novel_cats = categories - self.seen_categories
        is_novel = len(novel_cats) > 0

        self.last_was_novel = is_novel

        if is_novel:
            self.novel_total += 1
            if is_accepted:
                self.novel_accepted += 1
            else:
                self.novel_rejected += 1
        else:
            self.redundant_total += 1
            if is_accepted:
                self.redundant_accepted += 1
            else:
                self.redundant_rejected += 1

        # Update seen categories (regardless of accept/reject -- we track
        # what appeared in the stream, not what was trained on)
        self.seen_categories.update(categories)

    def get_stats(self) -> Dict[str, Any]:
        """Return novelty tracking statistics."""
        return {
            "categories_seen": len(self.seen_categories),
            "categories_list": sorted(self.seen_categories),
            "novel_total": self.novel_total,
            "novel_accepted": self.novel_accepted,
            "novel_rejected": self.novel_rejected,
            "novel_accept_rate": (
                self.novel_accepted / max(self.novel_total, 1)
            ),
            "redundant_total": self.redundant_total,
            "redundant_accepted": self.redundant_accepted,
            "redundant_rejected": self.redundant_rejected,
            "redundant_reject_rate": (
                self.redundant_rejected / max(self.redundant_total, 1)
            ),
            "empty_total": self.empty_total,
            "empty_accepted": self.empty_accepted,
        }

    def print_summary(self) -> None:
        stats = self.get_stats()
        print()
        print("=" * 60)
        print("Novelty Tracking Summary")
        print("=" * 60)
        print(f"  Categories seen    : {stats['categories_seen']}")
        print(f"  Novel frames       : {stats['novel_total']} "
              f"(accepted: {stats['novel_accepted']}, rejected: {stats['novel_rejected']})")
        print(f"  Novel accept rate  : {stats['novel_accept_rate']:.4f}")
        print(f"  Redundant frames   : {stats['redundant_total']} "
              f"(accepted: {stats['redundant_accepted']}, rejected: {stats['redundant_rejected']})")
        print(f"  Redundant rej rate : {stats['redundant_reject_rate']:.4f}")
        print(f"  Empty frames       : {stats['empty_total']}")
        print("=" * 60)
        print()
