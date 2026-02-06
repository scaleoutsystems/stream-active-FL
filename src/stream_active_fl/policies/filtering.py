"""
Filter policies for selective training in streaming learning.

Policies decide which stream items should trigger parameter updates (train),
be stored for replay (store), or be skipped entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
        return {"count_train": self.count_train}


class DifficultyBasedPolicy(FilterPolicy):
    """
    Difficulty-based selective training policy.

    Computes the loss on the stream item and triggers training only when:
    1. Loss exceeds a threshold (tau_loss), AND
    2. Teacher confidence exceeds a threshold (tau_teacher)

    This combines hard example mining with pseudo-label quality gating.

    Args:
        tau_loss: Loss threshold. Only train if loss > tau_loss.
        tau_teacher: Teacher confidence threshold. Only train if confidence >= tau_teacher.
        store_all: If True, store all items regardless of action. If False, only store
                   items that pass the filter (train items).
    """

    def __init__(
        self,
        tau_loss: float = 0.5,
        tau_teacher: float = 0.5,
        store_all: bool = False,
    ):
        self.tau_loss = tau_loss
        self.tau_teacher = tau_teacher
        self.store_all = store_all

        # Statistics
        self.count_train = 0
        self.count_store = 0
        self.count_skip = 0
        self.total_loss = 0.0
        self.num_loss_computed = 0

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ) -> Action:
        """
        Select action based on difficulty (loss) and teacher confidence.

        Note: This requires a forward pass to compute loss, but not a backward pass
        unless action is "train". The forward pass cost is unavoidable for difficulty-based
        filtering.
        """
        # Check teacher confidence first (no forward pass needed)
        if stream_item.teacher_score < self.tau_teacher:
            self.count_skip += 1
            if self.store_all:
                self.count_store += 1
                return "store"
            return "skip"

        # Compute loss (requires forward pass)
        model.eval()  # Don't affect batch norm stats
        with torch.no_grad():
            image = stream_item.image.unsqueeze(0).to(device)
            target = torch.tensor([stream_item.target], dtype=torch.float32, device=device)
            
            logits = model(image)
            loss = criterion(logits, target)
            loss_value = loss.item()

        self.total_loss += loss_value
        self.num_loss_computed += 1

        # Decide action based on loss threshold
        if loss_value > self.tau_loss:
            self.count_train += 1
            return "train"
        else:
            self.count_skip += 1
            if self.store_all:
                self.count_store += 1
                return "store"
            return "skip"

    def get_stats(self) -> Dict[str, Any]:
        avg_loss = self.total_loss / max(self.num_loss_computed, 1)
        total = self.count_train + self.count_store + self.count_skip
        return {
            "count_train": self.count_train,
            "count_store": self.count_store,
            "count_skip": self.count_skip,
            "train_rate": self.count_train / max(total, 1),
            "store_rate": self.count_store / max(total, 1),
            "skip_rate": self.count_skip / max(total, 1),
            "avg_loss": avg_loss,
            "tau_loss": self.tau_loss,
            "tau_teacher": self.tau_teacher,
        }


class TopKPolicy(FilterPolicy):
    """
    Top-K difficulty-based policy.

    Maintains a sliding window and trains on the top-K highest-loss items
    within each window.

    Args:
        window_size: Size of the sliding window.
        k: Number of items to train on per window.
        tau_teacher: Teacher confidence threshold (same as DifficultyBasedPolicy).
    """

    def __init__(
        self,
        window_size: int = 100,
        k: int = 10,
        tau_teacher: float = 0.5,
    ):
        self.window_size = window_size
        self.k = k
        self.tau_teacher = tau_teacher

        # Sliding window: list of (stream_item, loss_value)
        self.window: list = []

        # Statistics
        self.count_train = 0
        self.count_store = 0
        self.count_skip = 0

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ) -> Action:
        """
        Compute loss and add to window. Train if item is in top-K of current window.
        """
        # Check teacher confidence first
        if stream_item.teacher_score < self.tau_teacher:
            self.count_skip += 1
            return "skip"

        # Compute loss
        model.eval()
        with torch.no_grad():
            image = stream_item.image.unsqueeze(0).to(device)
            target = torch.tensor([stream_item.target], dtype=torch.float32, device=device)
            
            logits = model(image)
            loss = criterion(logits, target)
            loss_value = loss.item()

        # Add to window
        self.window.append((stream_item, loss_value))
        if len(self.window) > self.window_size:
            self.window.pop(0)

        # Check if current item is in top-K
        sorted_window = sorted(self.window, key=lambda x: x[1], reverse=True)
        top_k_losses = [x[1] for x in sorted_window[:self.k]]

        if loss_value in top_k_losses:
            self.count_train += 1
            return "train"
        else:
            self.count_skip += 1
            return "skip"

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_train + self.count_store + self.count_skip
        return {
            "count_train": self.count_train,
            "count_store": self.count_store,
            "count_skip": self.count_skip,
            "train_rate": self.count_train / max(total, 1),
            "window_size": self.window_size,
            "k": self.k,
            "tau_teacher": self.tau_teacher,
        }
