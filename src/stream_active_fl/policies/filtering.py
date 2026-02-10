"""
Filter policies for selective training in streaming learning.

Policies decide which stream items should trigger parameter updates (train),
be stored for replay (store), or be skipped entirely.

Available policies:
- NoFilterPolicy: Train on every item (unfiltered baseline)
- DifficultyBasedPolicy: Train on high-loss items, with optional adaptive
  thresholding (percentile-based)
- TopKPolicy: Train on top-K hardest items in a sliding window

All policies can be wrapped with TeacherConfidenceGate to additionally
filter out positive items with low-confidence pseudo-labels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn

from ..core.items import StreamItem


Action = Literal["train", "store", "skip"]


class FilterPolicy(ABC):
    """
    Base class for filter policies.

    A policy examines a stream item and the current model state to decide
    whether to train, store, or skip.
    """

    @abstractmethod
    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ) -> Action:
        """
        Select an action for the given stream item.

        Args:
            stream_item: The current stream item.
            model: The current model.
            criterion: Loss function.
            device: Device to run computations on.

        Returns:
            "train", "store", or "skip".
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return policy statistics (for logging)."""
        return {}


def _compute_item_loss(
    stream_item: StreamItem,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute loss for a single stream item (forward pass only, no grad)."""
    model.eval()
    with torch.no_grad():
        image = stream_item.image.unsqueeze(0).to(device)
        target = torch.tensor([stream_item.target], dtype=torch.float32, device=device)
        logits = model(image)
        loss = criterion(logits, target)
    return loss.item()


# =============================================================================
# TeacherConfidenceGate (general-purpose wrapper)
# =============================================================================


class TeacherConfidenceGate(FilterPolicy):
    """
    General-purpose wrapper that gates on teacher pseudo-label confidence.

    Wraps any FilterPolicy and skips positive items whose teacher confidence
    is below a threshold. This removes uncertain pseudo-labels from training.

    Only positive items (target == 1) are gated, because negative items
    (target == 0) have teacher_score == 0 by convention (no detection found),
    which is actually a confident prediction — the teacher is confident
    nothing is there.

    Args:
        inner_policy: The underlying filter policy to delegate to.
        tau_teacher: Confidence threshold. Positive items with
            teacher_score < tau_teacher are skipped.
        store_gated: If True, gated items get action "store" instead of "skip".
    """

    def __init__(
        self,
        inner_policy: FilterPolicy,
        tau_teacher: float = 0.5,
        store_gated: bool = False,
    ):
        self.inner_policy = inner_policy
        self.tau_teacher = tau_teacher
        self.store_gated = store_gated

        # Statistics for the gate itself
        self.count_gated = 0
        self.count_passed = 0

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ) -> Action:
        # Only gate positive items with low teacher confidence
        if (
            stream_item.target == 1.0
            and stream_item.teacher_score < self.tau_teacher
        ):
            self.count_gated += 1
            return "store" if self.store_gated else "skip"

        # Otherwise, delegate to inner policy
        self.count_passed += 1
        return self.inner_policy.select_action(stream_item, model, criterion, device)

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_gated + self.count_passed
        inner_stats = self.inner_policy.get_stats()

        gate_stats = {
            "teacher_gate_tau": self.tau_teacher,
            "teacher_gate_gated": self.count_gated,
            "teacher_gate_passed": self.count_passed,
            "teacher_gate_rate": self.count_gated / max(total, 1),
        }

        # Merge inner policy stats (inner stats take precedence for shared keys)
        return {**gate_stats, **inner_stats}


# =============================================================================
# NoFilterPolicy
# =============================================================================


class NoFilterPolicy(FilterPolicy):
    """
    Baseline policy: train on every stream item.

    This is the unfiltered streaming baseline.
    """

    def __init__(self):
        self.count_train = 0

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ) -> Action:
        self.count_train += 1
        return "train"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "count_train": self.count_train,
            "train_rate": 1.0,
        }


# =============================================================================
# DifficultyBasedPolicy
# =============================================================================


class DifficultyBasedPolicy(FilterPolicy):
    """
    Difficulty-based selective training policy.

    Computes the loss on each stream item (forward pass) and triggers training
    only for items that are sufficiently "hard". Supports two modes:

    **Adaptive mode (default)**: Maintains a sliding window of recent losses
    and trains on items whose loss falls in the top fraction. This avoids
    needing to hand-tune an absolute loss threshold.

    **Fixed mode**: Uses absolute thresholds (tau_loss). Only useful when you
    know the loss scale well.

    Args:
        adaptive: If True, use percentile-based adaptive thresholding.
        train_fraction: Fraction of items to train on (only if adaptive=True).
            E.g. 0.3 means train on the ~30% hardest items.
        loss_window_size: Size of the sliding window for loss history
            (only if adaptive=True).
        warmup_items: Number of items to process before applying the adaptive
            threshold. During warmup, all items are trained on to build up
            an initial loss distribution.
        tau_loss: Absolute loss threshold (only if adaptive=False).
        store_skipped: If True, skipped items are stored (action="store")
            instead of fully discarded.
    """

    def __init__(
        self,
        adaptive: bool = True,
        train_fraction: float = 0.3,
        loss_window_size: int = 500,
        warmup_items: int = 200,
        tau_loss: float = 0.5,
        store_skipped: bool = False,
    ):
        self.adaptive = adaptive
        self.train_fraction = train_fraction
        self.loss_window_size = loss_window_size
        self.warmup_items = warmup_items
        self.tau_loss = tau_loss
        self.store_skipped = store_skipped

        # Sliding window of recent losses (for adaptive mode)
        self.loss_history: deque = deque(maxlen=loss_window_size)

        # Statistics
        self.count_train = 0
        self.count_store = 0
        self.count_skip = 0
        self.items_seen = 0
        self.total_loss = 0.0

    def _get_adaptive_threshold(self) -> float:
        """
        Compute the adaptive loss threshold from the loss history.

        Returns the (1 - train_fraction) percentile of recent losses.
        E.g. if train_fraction=0.3, returns the 70th percentile, so items
        with loss above this value (top 30%) will be trained on.
        """
        if len(self.loss_history) == 0:
            return 0.0

        sorted_losses = sorted(self.loss_history)
        percentile_idx = int(len(sorted_losses) * (1.0 - self.train_fraction))
        percentile_idx = min(percentile_idx, len(sorted_losses) - 1)
        return sorted_losses[percentile_idx]

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ) -> Action:
        self.items_seen += 1

        # Compute loss (requires forward pass)
        loss_value = _compute_item_loss(stream_item, model, criterion, device)
        self.total_loss += loss_value
        self.loss_history.append(loss_value)

        # During warmup, train on everything (to build loss distribution)
        if self.adaptive and self.items_seen <= self.warmup_items:
            self.count_train += 1
            return "train"

        # Determine threshold
        if self.adaptive:
            threshold = self._get_adaptive_threshold()
        else:
            threshold = self.tau_loss

        # Decide action
        if loss_value > threshold:
            self.count_train += 1
            return "train"
        else:
            if self.store_skipped:
                self.count_store += 1
                return "store"
            self.count_skip += 1
            return "skip"

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_train + self.count_store + self.count_skip
        avg_loss = self.total_loss / max(len(self.loss_history), 1)

        stats = {
            "count_train": self.count_train,
            "count_store": self.count_store,
            "count_skip": self.count_skip,
            "train_rate": self.count_train / max(total, 1),
            "avg_loss": avg_loss,
            "items_seen": self.items_seen,
            "adaptive": self.adaptive,
        }

        if self.adaptive:
            stats["train_fraction"] = self.train_fraction
            stats["current_threshold"] = self._get_adaptive_threshold()
            stats["loss_window_size"] = len(self.loss_history)
        else:
            stats["tau_loss"] = self.tau_loss

        return stats


# =============================================================================
# TopKPolicy
# =============================================================================


class TopKPolicy(FilterPolicy):
    """
    Top-K difficulty-based policy.

    Maintains a sliding window and trains on items whose loss is among the
    top-K highest in the current window. This provides a relative (rather
    than absolute) selection criterion.

    Args:
        window_size: Size of the sliding window.
        k: Number of items to train on per window.
    """

    def __init__(
        self,
        window_size: int = 100,
        k: int = 30,
    ):
        self.window_size = window_size
        self.k = k

        # Sliding window: stores only loss values (not full items)
        self.loss_window: deque = deque(maxlen=window_size)

        # Statistics
        self.count_train = 0
        self.count_skip = 0
        self.items_seen = 0

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ) -> Action:
        self.items_seen += 1

        # Compute loss
        loss_value = _compute_item_loss(stream_item, model, criterion, device)

        # Add to window
        self.loss_window.append(loss_value)

        # During initial fill, train on everything
        if len(self.loss_window) < self.window_size:
            self.count_train += 1
            return "train"

        # Check if current loss is >= the K-th highest loss in window
        sorted_losses = sorted(self.loss_window, reverse=True)
        kth_loss = sorted_losses[min(self.k - 1, len(sorted_losses) - 1)]

        if loss_value >= kth_loss:
            self.count_train += 1
            return "train"
        else:
            self.count_skip += 1
            return "skip"

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_train + self.count_skip
        return {
            "count_train": self.count_train,
            "count_skip": self.count_skip,
            "train_rate": self.count_train / max(total, 1),
            "items_seen": self.items_seen,
            "window_size": self.window_size,
            "k": self.k,
        }
