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
        self.accept_fraction: float = 1.0
        self._recent_decisions: deque[int] = deque(maxlen=200)

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
        """Return the adaptive threshold from a sliding window of scores.

        The threshold is the (1 - accept_fraction) quantile of the
        recent scores.  Items scoring above this threshold are accepted,
        which mechanically targets an acceptance rate of
        accept_fraction.

        Args:
            score_history: Recent scores (sliding window).
            accept_fraction: Target fraction of items to accept.
        """
        if len(score_history) == 0:
            return 0.0
        sorted_scores = sorted(score_history)
        idx = int(len(sorted_scores) * (1.0 - accept_fraction))
        idx = min(idx, len(sorted_scores) - 1)
        return sorted_scores[idx]

    def _effective_accept_fraction(self) -> float:
        """Return the accept fraction adjusted by a proportional correction.

        Uses the recent post-warmup decision history to detect systematic
        deviation from the configured accept_fraction and compensate.
        The correction is stateless (pure function of the current window)
        so it cannot wind up.
        """
        if len(self._recent_decisions) < 50:
            return self.accept_fraction
        observed = sum(self._recent_decisions) / len(self._recent_decisions)
        error = self.accept_fraction - observed  # positive ⇒ under-accepting
        return max(0.05, min(0.95, self.accept_fraction + 0.5 * error))


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

    To isolate domain novelty from backbone training drift, all modes
    can use a frozen scoring model (a snapshot of the bootstrap model)
    for embedding computation.  When scoring_model is provided, the
    model passed to select_action is ignored for scoring -- only the
    frozen copy is used.  This keeps scores in a stable embedding space
    and makes distance from the bootstrap reference a genuine measure of
    domain novelty.

    Supports three budget modes:
    - "adaptive": (default) Sliding-window quantile thresholding that
      targets a fixed accept_fraction per window, with feedback
      correction.  Accept rate stays near accept_fraction regardless
      of domain.
    - "fixed_threshold": Threshold calibrated once from warmup scores
      and optionally tracked with a slow EMA.  When the stream enters
      a novel domain, scores are genuinely higher → more accepts; in a
      familiar domain, scores are lower → fewer accepts.
      Accept rate varies freely with stream novelty.
    - "global_budget": Same scoring and threshold as "fixed_threshold",
      plus a hard cap: total accepts ≤
      accept_fraction * total_stream_items.

    Args:
        bootstrap_mean: Mean embedding vector from bootstrap (1D tensor).
        bootstrap_cov: Covariance matrix from bootstrap (2D tensor).
            Required for mode="mahalanobis".
        bootstrap_count: Number of samples used to compute bootstrap_mean/cov.
            Used as prior weight when update_stats=True. If unknown (<=0),
            running-stat updates are disabled to avoid biased mean updates.
        mode: Distance computation mode.
        accept_fraction: Fraction of items to accept.  Used as the per-window
            target for "adaptive", and for computing the total budget in
            "global_budget" mode.  Ignored for "fixed_threshold".
        budget_mode: One of "adaptive", "fixed_threshold", "global_budget".
        score_window_size: Size of the sliding window for adaptive thresholding
            (used by "adaptive" mode).
        warmup_items: Accept all items unconditionally during warmup to
            build a score distribution.
        embedding_buffer_size: For mode="knn", how many recent embeddings
            to store.
        knn_k: For mode="knn", number of nearest neighbors.
        update_stats: Whether to update running mean with accepted embeddings.
            Covariance is kept fixed to bootstrap_cov for mahalanobis mode.
            Forced to False when a scoring_model is provided, since
            the reference must stay in the scoring model's embedding space.
        threshold_percentile: For "fixed_threshold" / "global_budget" modes,
            the percentile of warmup scores used to initialise the threshold.
            e.g. 0.5 = warmup median.
        threshold_ema_alpha: EMA smoothing factor for threshold tracking in
            "fixed_threshold" / "global_budget" modes.  0.0 = truly fixed
            threshold (strongest domain signal).  Increase (e.g. 0.0001)
            only if model-training-induced score-scale drift causes
            acceptance to collapse over long streams.
        total_stream_items: For "global_budget" mode, the expected total
            number of stream items (used to compute the accept budget).
        scoring_model: Optional frozen model for computing embeddings.
            When provided, select_action uses this model (not the live
            training model) so that scores remain in a stable embedding
            space.  The caller is responsible for freezing the model
            (eval mode, no grad).
        bootstrap_scores: Per-frame Mahalanobis distances from the
            bootstrap phase.  When provided for "fixed_threshold" /
            "global_budget" modes, the threshold is calibrated directly
            from these scores (no warmup period needed).  This makes the
            threshold independent of stream ordering.
    """

    def __init__(
        self,
        bootstrap_mean: torch.Tensor,
        bootstrap_cov: Optional[torch.Tensor] = None,
        bootstrap_count: int = 0,
        mode: Literal["mahalanobis", "cosine", "knn"] = "mahalanobis",
        accept_fraction: float = 0.3,
        budget_mode: Literal["adaptive", "fixed_threshold", "global_budget"] = "adaptive",
        score_window_size: int = 500,
        warmup_items: int = 100,
        embedding_buffer_size: int = 1000,
        knn_k: int = 10,
        update_stats: bool = True,
        threshold_percentile: float = 0.5,
        threshold_ema_alpha: float = 0.0,
        total_stream_items: int = 0,
        scoring_model: Optional[nn.Module] = None,
        bootstrap_scores: Optional[List[float]] = None,
    ):
        super().__init__()
        if mode == "mahalanobis" and bootstrap_cov is None:
            raise ValueError("mahalanobis mode requires bootstrap_cov")
        self.mode = mode
        self.accept_fraction = accept_fraction
        self.budget_mode = budget_mode
        self.score_window_size = score_window_size
        self.warmup_items = warmup_items
        self.knn_k = knn_k
        self.threshold_percentile = threshold_percentile
        self.threshold_ema_alpha = threshold_ema_alpha
        self.total_stream_items = total_stream_items

        # Running statistics
        self.mean = bootstrap_mean.clone().float()
        self.cov = bootstrap_cov.clone().float() if bootstrap_cov is not None else None
        self._cov_inv: Optional[torch.Tensor] = None

        if self.cov is not None:
            reg = 1e-5 * torch.eye(self.cov.shape[0])
            self._cov_inv = torch.linalg.inv(self.cov + reg)

        # Embedding buffer for kNN
        self.embedding_buffer: deque = deque(maxlen=embedding_buffer_size)

        # Sliding window for adaptive thresholding (used by "adaptive" mode)
        self.score_history: deque = deque(maxlen=score_window_size)

        # Counters
        self.items_seen = 0
        self.count_accept = 0
        self.count_reject = 0

        # Frozen scoring model -- if provided, all embeddings for scoring
        # are computed through this model instead of the live training model.
        self._scoring_model = scoring_model

        # Running-mean update state (prior from bootstrap stats).
        # Forced OFF when a scoring_model is provided (the reference must
        # stay in the scoring model's embedding space).
        self.bootstrap_count = int(max(bootstrap_count, 0))
        if scoring_model is not None:
            self.update_stats = False
        else:
            self.update_stats = bool(update_stats and self.bootstrap_count > 0)
        self._n_embeddings = self.bootstrap_count

        # EMA threshold for fixed_threshold / global_budget modes.
        self._ema_threshold: float = 0.0
        self._bootstrap_calibrated: bool = False

        if (
            bootstrap_scores is not None
            and len(bootstrap_scores) > 0
            and budget_mode in ("fixed_threshold", "global_budget")
        ):
            sorted_scores = sorted(bootstrap_scores)
            idx = int(len(sorted_scores) * (1.0 - threshold_percentile))
            idx = min(idx, len(sorted_scores) - 1)
            self._ema_threshold = sorted_scores[idx]
            self._bootstrap_calibrated = True

        # Global budget (for "global_budget" mode)
        self._budget_remaining: int = max(
            1, int(accept_fraction * total_stream_items)
        ) if budget_mode == "global_budget" and total_stream_items > 0 else 0

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
        if self.budget_mode in ("fixed_threshold", "global_budget"):
            return self._ema_threshold
        return self._compute_adaptive_threshold(
            self.score_history, self._effective_accept_fraction(),
        )

    def _calibrate_ema_threshold(self) -> None:
        """Initialise the EMA threshold from warmup scores."""
        if not self.score_history:
            self._ema_threshold = 0.0
            return
        sorted_scores = sorted(self.score_history)
        idx = int(len(sorted_scores) * (1.0 - self.threshold_percentile))
        idx = min(idx, len(sorted_scores) - 1)
        self._ema_threshold = sorted_scores[idx]

    def _update_ema_threshold(self, score: float) -> None:
        """Slowly track score-scale drift without chasing domain content."""
        self._ema_threshold += self.threshold_ema_alpha * (score - self._ema_threshold)

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
        scorer = self._scoring_model if self._scoring_model is not None else model
        embedding = scorer.get_embedding([image]).squeeze(0).cpu()

        score = self._compute_score(embedding)
        self.score_history.append(score)

        if self.update_stats:
            self._update_running_stats(embedding)

        meta: Dict[str, Any] = {"score": score, "mode": self.mode, "budget_mode": self.budget_mode}

        # Warmup: accept all to build score distribution.
        # Skipped when threshold was pre-calibrated from bootstrap scores.
        if not self._bootstrap_calibrated and self.items_seen <= self.warmup_items:
            self.count_accept += 1
            if self.budget_mode == "global_budget":
                self._budget_remaining -= 1
            self.selection_tracker.record("accept", stream_item.categories)
            if self.items_seen == self.warmup_items and self.budget_mode in ("fixed_threshold", "global_budget"):
                self._calibrate_ema_threshold()
            return ("accept", meta)

        # Slowly track score-scale drift for EMA-threshold modes
        if self.budget_mode in ("fixed_threshold", "global_budget"):
            self._update_ema_threshold(score)

        threshold = self._get_threshold()
        meta["threshold"] = threshold

        # Global budget: reject everything once budget exhausted
        if self.budget_mode == "global_budget" and self._budget_remaining <= 0:
            self.count_reject += 1
            self._recent_decisions.append(0)
            self.selection_tracker.record("reject", stream_item.categories)
            return ("reject", meta)

        if score >= threshold:
            self.count_accept += 1
            self._recent_decisions.append(1)
            self.selection_tracker.record("accept", stream_item.categories)
            if self.budget_mode == "global_budget":
                self._budget_remaining -= 1
            return ("accept", meta)
        else:
            self.count_reject += 1
            self._recent_decisions.append(0)
            self.selection_tracker.record("reject", stream_item.categories)
            return ("reject", meta)

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_accept + self.count_reject
        stats: Dict[str, Any] = {
            "count_accept": self.count_accept,
            "count_reject": self.count_reject,
            "accept_rate": self.count_accept / max(total, 1),
            "items_seen": self.items_seen,
            "mode": self.mode,
            "budget_mode": self.budget_mode,
            "accept_fraction": self.accept_fraction,
            "bootstrap_count": self.bootstrap_count,
            "update_stats_enabled": self.update_stats,
            "frozen_scoring_model": self._scoring_model is not None,
            "current_threshold": self._get_threshold(),
            "bootstrap_calibrated": self._bootstrap_calibrated,
        }
        if self.budget_mode == "adaptive":
            stats["effective_accept_fraction"] = self._effective_accept_fraction()
        if self.budget_mode in ("fixed_threshold", "global_budget"):
            stats["threshold_percentile"] = self.threshold_percentile
            stats["threshold_ema_alpha"] = self.threshold_ema_alpha
        if self.budget_mode == "global_budget":
            stats["budget_remaining"] = self._budget_remaining
        return stats


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

        top_scores = scores.sort(descending=True).values[: self.top_k_detections]
        mean_conf = float(top_scores.mean().item())
        return 1.0 - mean_conf

    def _get_threshold(self) -> float:
        return self._compute_adaptive_threshold(
            self.score_history, self._effective_accept_fraction(),
        )

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
            self._recent_decisions.append(1)
            self.selection_tracker.record("accept", stream_item.categories)
            return ("accept", meta)
        else:
            self.count_reject += 1
            self._recent_decisions.append(0)
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
            "effective_accept_fraction": self._effective_accept_fraction(),
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
        return self._compute_adaptive_threshold(
            self.norm_history, self._effective_accept_fraction(),
        )

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
            self._recent_decisions.append(1)
            self.selection_tracker.record("accept", stream_item.categories)
            return ("accept", meta)
        else:
            self.count_reject += 1
            self._recent_decisions.append(0)
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
            "effective_accept_fraction": self._effective_accept_fraction(),
            "current_threshold": self._get_threshold(),
        }
