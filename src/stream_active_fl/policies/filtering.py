"""
Filter policies for selective training in buffer-based streaming learning.

Policies decide which stream items should be accepted (added to the training
buffer) or rejected (discarded).

Available policies:
- NoFilterPolicy: Accept every item (unfiltered baseline)
- RandomPolicy: Accept each item with fixed probability (random baseline)
- DistributionBasedPolicy: Accept items whose backbone embedding falls on the
  tail of the distribution seen so far (novel / unusual frames)
- UncertaintyBasedPolicy: Accept items where the model's detection confidence
  is low (the model is uncertain = frame is informative)
- GradientNormPolicy: Accept items with the largest parameter gradient norms

Each policy returns ("accept" | "reject", metadata_dict) where the metadata
dict carries information for logging (e.g. the score used for the decision).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from ..core.items import StreamItem


Action = Literal["accept", "reject"]
FilterResult = Tuple[Action, Dict[str, Any]]


# =============================================================================
# Selection tracking
# =============================================================================


@dataclass
class SelectionTracker:
    """
    Tracks per-category, per-action selection statistics over an interval.

    Records what the filter selects (accept/reject) broken down by the
    categories present in each frame.  Stats accumulate between calls to
    reset_interval(), giving per-checkpoint-interval visibility.
    """

    accept_count: int = 0
    reject_count: int = 0
    accept_by_category: Dict[str, int] = field(default_factory=dict)
    reject_by_category: Dict[str, int] = field(default_factory=dict)

    def record(self, action: Action, categories: set[str]) -> None:
        """Record a single filter decision."""
        if action == "accept":
            self.accept_count += 1
            target = self.accept_by_category
        else:
            self.reject_count += 1
            target = self.reject_by_category

        for cat in categories:
            target[cat] = target.get(cat, 0) + 1

    def get_interval_stats(self) -> Dict[str, Any]:
        total = self.accept_count + self.reject_count
        return {
            "accept_count": self.accept_count,
            "reject_count": self.reject_count,
            "total": total,
            "accept_rate": self.accept_count / max(total, 1),
            "accept_by_category": dict(self.accept_by_category),
            "reject_by_category": dict(self.reject_by_category),
        }

    def reset_interval(self) -> None:
        self.accept_count = 0
        self.reject_count = 0
        self.accept_by_category = {}
        self.reject_by_category = {}


# =============================================================================
# Base class
# =============================================================================


class FilterPolicy(ABC):
    """
    Base class for filter policies.

    A policy examines a stream item and the current model state to decide
    whether to accept (add to training buffer) or reject (discard).
    """

    def __init__(self):
        self.selection_tracker = SelectionTracker()

    @abstractmethod
    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        device: torch.device,
    ) -> FilterResult:
        """
        Decide whether to accept or reject the given stream item.

        Args:
            stream_item: The current stream item.
            model: The current model (may be used for embedding / uncertainty).
            device: Device to run computations on.

        Returns:
            (action, metadata): action is "accept" or "reject".
            metadata is a dict with policy-specific info for logging
            (e.g. {"score": 0.42, "threshold": 0.35}).
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Return policy statistics (for logging)."""
        return {}

    def get_selection_stats(self) -> Dict[str, Any]:
        return self.selection_tracker.get_interval_stats()

    def reset_selection_stats(self) -> None:
        self.selection_tracker.reset_interval()

    def requires_model_forward(self) -> bool:
        """Whether this policy needs model inference/gradients per item."""
        return True

    @staticmethod
    def _compute_adaptive_threshold(
        score_history: deque,
        accept_fraction: float,
    ) -> float:
        """Return the adaptive threshold from a sliding window of scores."""
        if len(score_history) == 0:
            return 0.0
        sorted_scores = sorted(score_history)
        idx = int(len(sorted_scores) * (1.0 - accept_fraction))
        idx = min(idx, len(sorted_scores) - 1)
        return sorted_scores[idx]


# =============================================================================
# NoFilterPolicy
# =============================================================================


class NoFilterPolicy(FilterPolicy):
    """Baseline policy: accept every stream item."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        device: torch.device,
    ) -> FilterResult:
        self.count += 1
        self.selection_tracker.record("accept", stream_item.categories)
        return ("accept", {})

    def get_stats(self) -> Dict[str, Any]:
        return {"count_accept": self.count, "accept_rate": 1.0}

    def requires_model_forward(self) -> bool:
        return False


# =============================================================================
# RandomPolicy
# =============================================================================


class RandomPolicy(FilterPolicy):
    """
    Random baseline: accept each item with fixed probability.

    Accepts items with probability accept_fraction, independent of content.
    Used to compare whether content-based filtering outperforms random selection.

    Args:
        accept_fraction: Probability of accepting each item (e.g. 0.3 = 30%).
    """

    def __init__(self, accept_fraction: float = 0.3):
        super().__init__()
        self.accept_fraction = accept_fraction
        self.count_accept = 0
        self.count_reject = 0

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        device: torch.device,
    ) -> FilterResult:
        r = random.random()
        is_accepted = r < self.accept_fraction
        if is_accepted:
            self.count_accept += 1
            self.selection_tracker.record("accept", stream_item.categories)
            return ("accept", {"random_score": r})
        else:
            self.count_reject += 1
            self.selection_tracker.record("reject", stream_item.categories)
            return ("reject", {"random_score": r})

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_accept + self.count_reject
        return {
            "count_accept": self.count_accept,
            "count_reject": self.count_reject,
            "accept_rate": self.count_accept / max(total, 1),
            "accept_fraction": self.accept_fraction,
        }

    def requires_model_forward(self) -> bool:
        return False


# =============================================================================
# DistributionBasedPolicy
# =============================================================================


class DistributionBasedPolicy(FilterPolicy):
    """
    Embedding-distribution-based selective training policy.

    Maintains a running estimate of the embedding distribution seen so far
    (initialized from bootstrap statistics). For each new frame, extracts
    its backbone embedding and computes a distance score.  Frames on the
    "tail" of the distribution (high distance) are accepted as novel;
    frames near the center are rejected as redundant.

    Supports three distance modes:
    - "mahalanobis": Mahalanobis distance to the running mean (requires
      covariance from bootstrap).
    - "cosine": 1 - cosine_similarity to the running mean.
    - "knn": Average distance to the k nearest neighbors in a stored
      buffer of recent embeddings.

    Args:
        bootstrap_mean: Mean embedding vector from bootstrap (1D tensor).
        bootstrap_cov: Covariance matrix from bootstrap (2D tensor).
            Required for mode="mahalanobis".
        bootstrap_count: Number of samples used to compute bootstrap_mean/cov.
            Used as prior weight when update_stats=True. If unknown (<=0),
            running-stat updates are disabled to avoid biased mean updates.
        mode: Distance computation mode.
        accept_fraction: Fraction of items to accept (top percentile by
            distance). E.g. 0.3 means accept the ~30% most distant items.
        score_window_size: Size of the sliding window for adaptive thresholding.
        warmup_items: Accept all items unconditionally during warmup to
            build a score distribution.
        embedding_buffer_size: For mode="knn", how many recent embeddings
            to store.
        knn_k: For mode="knn", number of nearest neighbors.
        update_stats: Whether to update running mean with accepted embeddings.
            Covariance is kept fixed to bootstrap_cov for mahalanobis mode.
    """

    def __init__(
        self,
        bootstrap_mean: torch.Tensor,
        bootstrap_cov: Optional[torch.Tensor] = None,
        bootstrap_count: int = 0,
        mode: Literal["mahalanobis", "cosine", "knn"] = "mahalanobis",
        accept_fraction: float = 0.3,
        score_window_size: int = 500,
        warmup_items: int = 100,
        embedding_buffer_size: int = 1000,
        knn_k: int = 10,
        update_stats: bool = True,
    ):
        super().__init__()
        if mode == "mahalanobis" and bootstrap_cov is None:
            raise ValueError("mahalanobis mode requires bootstrap_cov")
        self.mode = mode
        self.accept_fraction = accept_fraction
        self.score_window_size = score_window_size
        self.warmup_items = warmup_items
        self.knn_k = knn_k

        # Running statistics
        self.mean = bootstrap_mean.clone().float()
        self.cov = bootstrap_cov.clone().float() if bootstrap_cov is not None else None
        self._cov_inv: Optional[torch.Tensor] = None

        if self.cov is not None:
            reg = 1e-5 * torch.eye(self.cov.shape[0])
            self._cov_inv = torch.linalg.inv(self.cov + reg)

        # Embedding buffer for kNN
        self.embedding_buffer: deque = deque(maxlen=embedding_buffer_size)

        # Sliding window for adaptive thresholding
        self.score_history: deque = deque(maxlen=score_window_size)

        # Counters
        self.items_seen = 0
        self.count_accept = 0
        self.count_reject = 0

        # Running-mean update state (prior from bootstrap stats).
        self.bootstrap_count = int(max(bootstrap_count, 0))
        self.update_stats = bool(update_stats and self.bootstrap_count > 0)
        self._n_embeddings = self.bootstrap_count

    def _compute_score(self, embedding: torch.Tensor) -> float:
        """Compute a distance score for a single embedding (1D tensor)."""
        emb = embedding.float()

        if self.mode == "mahalanobis":
            if self._cov_inv is None:
                diff = emb - self.mean
                return float(diff.norm().item())
            diff = emb - self.mean
            return float(torch.sqrt(diff @ self._cov_inv @ diff).item())

        elif self.mode == "cosine":
            sim = torch.nn.functional.cosine_similarity(
                emb.unsqueeze(0), self.mean.unsqueeze(0)
            )
            return float(1.0 - sim.item())

        elif self.mode == "knn":
            if len(self.embedding_buffer) < self.knn_k:
                return float("inf")
            buf = torch.stack(list(self.embedding_buffer))
            dists = torch.cdist(emb.unsqueeze(0), buf).squeeze(0)
            topk_dists, _ = dists.topk(self.knn_k, largest=False)
            return float(topk_dists.mean().item())

        raise ValueError(f"Unknown mode: {self.mode}")

    def _get_threshold(self) -> float:
        return self._compute_adaptive_threshold(self.score_history, self.accept_fraction)

    def _update_running_stats(self, embedding: torch.Tensor) -> None:
        """Incrementally update running mean with a new embedding."""
        self._n_embeddings += 1
        emb = embedding.float()
        self.mean = self.mean + (emb - self.mean) / self._n_embeddings
        if self.mode == "knn":
            self.embedding_buffer.append(emb.cpu())

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        device: torch.device,
    ) -> FilterResult:
        self.items_seen += 1

        image = stream_item.image.to(device)
        embedding = model.get_embedding([image]).squeeze(0).cpu()

        score = self._compute_score(embedding)
        self.score_history.append(score)

        meta = {"score": score, "mode": self.mode}

        # Warmup: accept all to build score distribution
        if self.items_seen <= self.warmup_items:
            self.count_accept += 1
            if self.update_stats:
                self._update_running_stats(embedding)
            self.selection_tracker.record("accept", stream_item.categories)
            return ("accept", meta)

        threshold = self._get_threshold()
        meta["threshold"] = threshold

        if score >= threshold:
            self.count_accept += 1
            if self.update_stats:
                self._update_running_stats(embedding)
            self.selection_tracker.record("accept", stream_item.categories)
            return ("accept", meta)
        else:
            self.count_reject += 1
            self.selection_tracker.record("reject", stream_item.categories)
            return ("reject", meta)

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_accept + self.count_reject
        return {
            "count_accept": self.count_accept,
            "count_reject": self.count_reject,
            "accept_rate": self.count_accept / max(total, 1),
            "items_seen": self.items_seen,
            "mode": self.mode,
            "accept_fraction": self.accept_fraction,
            "bootstrap_count": self.bootstrap_count,
            "update_stats_enabled": self.update_stats,
            "current_threshold": self._get_threshold(),
        }


# =============================================================================
# UncertaintyBasedPolicy
# =============================================================================


class UncertaintyBasedPolicy(FilterPolicy):
    """
    Prediction-uncertainty-based selective training policy.

    Runs inference on each frame and measures prediction uncertainty:
    the fewer high-confidence detections the model produces, the more
    uncertain it is about the frame content.

    Uncertainty score = 1 - (mean of top-K detection scores).
    Frames where the model is most uncertain are accepted.

    Args:
        accept_fraction: Fraction of items to accept (highest uncertainty).
        score_window_size: Size of the sliding window for adaptive thresholding.
        warmup_items: Accept all items unconditionally during warmup.
        confidence_threshold: Minimum detection score to consider.
        top_k_detections: How many top detections to average for the
            confidence estimate.
    """

    def __init__(
        self,
        accept_fraction: float = 0.3,
        score_window_size: int = 500,
        warmup_items: int = 100,
        confidence_threshold: float = 0.1,
        top_k_detections: int = 5,
    ):
        super().__init__()
        self.accept_fraction = accept_fraction
        self.score_window_size = score_window_size
        self.warmup_items = warmup_items
        self.confidence_threshold = confidence_threshold
        self.top_k_detections = top_k_detections

        self.score_history: deque = deque(maxlen=score_window_size)
        self.items_seen = 0
        self.count_accept = 0
        self.count_reject = 0

    def _compute_uncertainty(self, predictions: Dict[str, torch.Tensor]) -> float:
        """Compute uncertainty from detection predictions."""
        scores = predictions["scores"]
        scores = scores[scores >= self.confidence_threshold]

        if len(scores) == 0:
            return 1.0  # max uncertainty: no detections at all

        top_scores = scores[: self.top_k_detections]
        mean_conf = float(top_scores.mean().item())
        return 1.0 - mean_conf

    def _get_threshold(self) -> float:
        return self._compute_adaptive_threshold(self.score_history, self.accept_fraction)

    @torch.no_grad()
    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        device: torch.device,
    ) -> FilterResult:
        self.items_seen += 1

        was_training = model.training
        model.eval()

        image = stream_item.image.to(device)
        predictions = model([image])[0]

        if was_training:
            model.train()

        uncertainty = self._compute_uncertainty(predictions)
        self.score_history.append(uncertainty)

        meta = {"uncertainty": uncertainty}

        if self.items_seen <= self.warmup_items:
            self.count_accept += 1
            self.selection_tracker.record("accept", stream_item.categories)
            return ("accept", meta)

        threshold = self._get_threshold()
        meta["threshold"] = threshold

        if uncertainty >= threshold:
            self.count_accept += 1
            self.selection_tracker.record("accept", stream_item.categories)
            return ("accept", meta)
        else:
            self.count_reject += 1
            self.selection_tracker.record("reject", stream_item.categories)
            return ("reject", meta)

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_accept + self.count_reject
        return {
            "count_accept": self.count_accept,
            "count_reject": self.count_reject,
            "accept_rate": self.count_accept / max(total, 1),
            "items_seen": self.items_seen,
            "accept_fraction": self.accept_fraction,
            "current_threshold": self._get_threshold(),
        }


# =============================================================================
# GradientNormPolicy
# =============================================================================


class GradientNormPolicy(FilterPolicy):
    """
    Gradient-norm selective training policy.

    Computes the L2 norm of per-sample parameter gradients as an importance
    score and selects items with the highest norms for training.  Gradient
    norm captures how much a sample would move the model parameters.

    Args:
        accept_fraction: Fraction of items to accept (highest gradient norm).
        norm_window_size: Sliding window for adaptive thresholding.
        warmup_items: Accept all items during warmup.
    """

    def __init__(
        self,
        accept_fraction: float = 0.3,
        norm_window_size: int = 500,
        warmup_items: int = 200,
    ):
        super().__init__()
        self.accept_fraction = accept_fraction
        self.norm_window_size = norm_window_size
        self.warmup_items = warmup_items

        self.norm_history: deque = deque(maxlen=norm_window_size)
        self.items_seen = 0
        self.count_accept = 0
        self.count_reject = 0

    def _compute_gradient_norm(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        device: torch.device,
    ) -> float:
        was_training = model.training
        model.train()
        image = stream_item.image.to(device)
        target = {
            "boxes": stream_item.annotations["boxes"].to(device),
            "labels": stream_item.annotations["labels"].to(device),
        }
        loss_dict = model([image], [target])
        loss = torch.stack(list(loss_dict.values())).sum()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            loss, trainable_params, retain_graph=False, allow_unused=True,
        )
        total = 0.0
        for g in grads:
            if g is not None:
                total += g.pow(2).sum().item()

        if not was_training:
            model.eval()

        return total ** 0.5

    def _get_threshold(self) -> float:
        return self._compute_adaptive_threshold(self.norm_history, self.accept_fraction)

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        device: torch.device,
    ) -> FilterResult:
        self.items_seen += 1

        grad_norm = self._compute_gradient_norm(stream_item, model, device)
        self.norm_history.append(grad_norm)

        meta = {"grad_norm": grad_norm}

        if self.items_seen <= self.warmup_items:
            self.count_accept += 1
            self.selection_tracker.record("accept", stream_item.categories)
            return ("accept", meta)

        threshold = self._get_threshold()
        meta["threshold"] = threshold

        if grad_norm >= threshold:
            self.count_accept += 1
            self.selection_tracker.record("accept", stream_item.categories)
            return ("accept", meta)
        else:
            self.count_reject += 1
            self.selection_tracker.record("reject", stream_item.categories)
            return ("reject", meta)

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_accept + self.count_reject
        return {
            "count_accept": self.count_accept,
            "count_reject": self.count_reject,
            "accept_rate": self.count_accept / max(total, 1),
            "items_seen": self.items_seen,
            "accept_fraction": self.accept_fraction,
            "current_threshold": self._get_threshold(),
        }
