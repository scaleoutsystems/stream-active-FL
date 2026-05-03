"""
Filter policies for selective training in buffer-based streaming learning.

Policies decide which stream items should be accepted (added to the training
buffer) or rejected (discarded).

Available policies:
- NoFilterPolicy: Accept every item (unfiltered baseline)
- RandomPolicy: Accept each item with fixed probability (random baseline)
- DistributionBasedPolicy: Accept items whose backbone Mahalanobis distance to
  the bootstrap-calibrated reference exceeds the threshold.  Optionally
  adaptive: the reference can be refreshed periodically from bootstrap plus
  either the last M accepted frames (sliding window) or a uniform random
  reservoir of size R over all past accepts.
- DetectionUncertaintyPolicy: Accept items whose detection-head uncertainty
  exceeds a bootstrap-calibrated threshold.  Two reductions supported
  (topk_mean or margin on post-NMS box confidences).  Optionally adaptive
  via the same refresh mechanism as DistributionBasedPolicy.
- MixturePolicy: Wraps a signal-based inner policy and mixes it with a
  random-acceptance fallback via epsilon-greedy routing (mixture_gamma
  fraction of frames follow the inner policy, the rest follow random).
  Addresses the sampling-bias failure mode of pure uncertainty/novelty
  selection while preserving a domain-aware signal.

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

    @abstractmethod
    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        device: torch.device,
    ) -> FilterResult:
        """
        Decide whether to accept or reject the given stream item.

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
    Mahalanobis-distance filter with a bootstrap-calibrated threshold.

    For each new frame the frozen scoring model produces a backbone embedding,
    the Mahalanobis distance to the reference (bootstrap) distribution is
    computed, and the frame is accepted when its distance is at or above the
    threshold.  The threshold is calibrated once from the per-frame distances
    of the bootstrap frames at threshold_percentile (e.g. 0.10 accepts
    the top 10% most-distant bootstrap frames -- and, by extension, any
    stream frame more distant than that cutoff).

    Accept rate therefore varies freely with stream novelty: in a familiar
    domain scores are low and acceptance falls; in a novel domain scores are
    high and acceptance rises.  That is the signal we care about.

    Adaptive refresh:
        An external ScoringRefresher can periodically re-embed
        (bootstrap + accepted-frame reference) through a fresh snapshot of
        the live training model and call apply_refresh to replace the
        scoring model, the mean, the covariance, and the threshold.  The
        accepted-frame reference is populated in one of two mutually
        exclusive modes:

        refresh_window_size = M > 0: deque of the last M accepted frame
        ids (FIFO sliding window).

        reservoir_size = R > 0: uniform random sample of all past accepts
        maintained with Vitter's Algorithm R.  Each of the first R accepts
        enters the reservoir unconditionally; for accept number N > R, a
        random index j is drawn from {0, ..., N-1} and, if j < R, the new
        frame replaces slot j.  Invariant: the reservoir is an unbiased
        uniform sample of size min(R, count_accept) over the entire
        accept history.

        With both sizes set to 0 the filter is static: no accepted frames
        participate in the reference, and a refresh only updates the
        scoring model (no change to mean/cov/threshold).

    Args:
        bootstrap_mean: Mean embedding vector from bootstrap (1D tensor).
        bootstrap_cov: Covariance matrix from bootstrap (2D tensor).
        accept_fraction: Nominal accept rate (only used for logging).
        threshold_percentile: Fraction of the reference distribution that
            should fall at or above the threshold.  E.g. 0.10 calibrates
            the threshold at the 90th percentile of reference distances.
        scoring_model: Frozen model (eval mode, no grad) used for embedding
            computation.  The model passed to select_action is ignored.
        bootstrap_scores: Per-frame Mahalanobis distances of the bootstrap
            frames (used to calibrate the threshold).
        refresh_window_size: If > 0, maintain a deque of the last M
            accepted frame ids.  Mutually exclusive with reservoir_size.
        reservoir_size: If > 0, maintain a uniform random reservoir of R
            accepted frame ids (Algorithm R).  Mutually exclusive with
            refresh_window_size.
        reservoir_seed: Seed for the reservoir's internal random.Random.
            Required when reservoir_size > 0 for reproducibility.
    """

    def __init__(
        self,
        bootstrap_mean: torch.Tensor,
        bootstrap_cov: torch.Tensor,
        *,
        scoring_model: nn.Module,
        bootstrap_scores: List[float],
        threshold_percentile: float = 0.10,
        accept_fraction: float = 0.10,
        refresh_window_size: int = 0,
        reservoir_size: int = 0,
        reservoir_seed: Optional[int] = None,
    ):
        super().__init__()
        if bootstrap_cov is None:
            raise ValueError("DistributionBasedPolicy requires bootstrap_cov")
        if scoring_model is None:
            raise ValueError("DistributionBasedPolicy requires a frozen scoring_model")
        if not bootstrap_scores:
            raise ValueError(
                "DistributionBasedPolicy requires bootstrap_scores to calibrate "
                "the threshold; regenerate bootstrap embeddings with scores."
            )

        refresh_window_size = max(0, int(refresh_window_size))
        reservoir_size = max(0, int(reservoir_size))
        if refresh_window_size > 0 and reservoir_size > 0:
            raise ValueError(
                "DistributionBasedPolicy: refresh_window_size and "
                "reservoir_size are mutually exclusive; set at most one > 0."
            )

        self.accept_fraction = accept_fraction
        self.threshold_percentile = threshold_percentile

        self.mean = bootstrap_mean.clone().float()
        self.cov = bootstrap_cov.clone().float()
        reg = 1e-5 * torch.eye(self.cov.shape[0])
        self._cov_inv: torch.Tensor = torch.linalg.inv(self.cov + reg)

        # Optional second Gaussian for two-reference (multimodal) scoring.
        # When set, _compute_score returns min(d_primary, d_secondary), so a
        # frame is "novel" only if it is far from BOTH known modes.  Populated
        # by apply_refresh when scoring_reference_mode == "two_reference"; left
        # None for the default single-reference behavior.
        self.mean2: Optional[torch.Tensor] = None
        self.cov2: Optional[torch.Tensor] = None
        self._cov_inv2: Optional[torch.Tensor] = None

        self._scoring_model = scoring_model
        self._threshold: float = self._percentile(bootstrap_scores, threshold_percentile)

        self.items_seen = 0
        self.count_accept = 0
        self.count_reject = 0

        self.refresh_window_size: int = refresh_window_size
        self.reservoir_size: int = reservoir_size
        self._accepted_deque: deque[str] = deque(
            maxlen=self.refresh_window_size if self.refresh_window_size > 0 else 1,
        )
        self._reservoir: List[str] = []
        self._reservoir_rng: random.Random = random.Random(reservoir_seed)
        self.num_refreshes: int = 0

    @staticmethod
    def _percentile(scores: List[float] | torch.Tensor, percentile: float) -> float:
        """Return the (1 - percentile) quantile of scores (top-p cutoff)."""
        if isinstance(scores, torch.Tensor):
            sorted_scores, _ = torch.sort(scores.float())
            arr = sorted_scores.tolist()
        else:
            arr = sorted(scores)
        if not arr:
            return 0.0
        idx = int(len(arr) * (1.0 - percentile))
        idx = min(max(idx, 0), len(arr) - 1)
        return float(arr[idx])

    def _compute_score(self, embedding: torch.Tensor) -> float:
        """Mahalanobis distance from embedding to the reference distribution.

        Single-reference mode: distance to the (mean, cov) Gaussian.

        Two-reference mode (mean2/cov2 set): min(d_primary, d_secondary),
        where the primary Gaussian is fitted on bootstrap frames and the
        secondary on the accepted-frame window/reservoir.  A frame is
        "novel" only when it is far from BOTH known modes, which avoids
        the unimodal-fit pathology where a fat single Gaussian over a
        bimodal reference set treats inter-mode points as familiar.
        """
        emb = embedding.float()
        diff1 = emb - self.mean
        d1 = float(torch.sqrt(diff1 @ self._cov_inv @ diff1).item())
        if self.mean2 is None or self._cov_inv2 is None:
            return d1
        diff2 = emb - self.mean2
        d2 = float(torch.sqrt(diff2 @ self._cov_inv2 @ diff2).item())
        return min(d1, d2)

    def _get_threshold(self) -> float:
        return self._threshold

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        device: torch.device,
    ) -> FilterResult:
        self.items_seen += 1

        image = stream_item.image.to(device)
        embedding = self._scoring_model.get_embedding([image]).squeeze(0).cpu()

        score = self._compute_score(embedding)
        threshold = self._threshold

        meta: Dict[str, Any] = {"score": score, "threshold": threshold}

        if score >= threshold:
            self.count_accept += 1
            self.selection_tracker.record("accept", stream_item.categories)
            self._record_accepted(stream_item)
            return ("accept", meta)

        self.count_reject += 1
        self.selection_tracker.record("reject", stream_item.categories)
        return ("reject", meta)

    def _record_accepted(self, stream_item: StreamItem) -> None:
        """Record an accepted frame-id in the active reference set (if any).

        Dispatches on the configured mode:
            refresh_window_size > 0: append to the FIFO deque; deque
            auto-evicts the oldest id once maxlen is reached.

            reservoir_size > 0: Vitter's Algorithm R.  count_accept has
            already been incremented in select_action before this is
            called, so it equals N (the 1-indexed accept number).  For
            N <= R: append.  For N > R: draw j ~ Uniform{0..N-1}; if
            j < R, replace _reservoir[j] with the new frame_id.

            Neither set: no-op.

        Frames without a frame_id in metadata are silently skipped so the
        refresher can't accidentally request entries for them.
        """
        frame_id = stream_item.metadata.get("frame_id")
        if frame_id is None:
            return
        fid = str(frame_id)

        if self.refresh_window_size > 0:
            self._accepted_deque.append(fid)
            return

        if self.reservoir_size > 0:
            n = self.count_accept  # 1-indexed: this is the Nth accept.
            r = self.reservoir_size
            if n <= r:
                self._reservoir.append(fid)
            else:
                j = self._reservoir_rng.randrange(n)  # [0, n-1]
                if j < r:
                    self._reservoir[j] = fid

    def get_accepted_frame_ids(self) -> List[str]:
        """Return a snapshot of the current accepted-frame reference.

        Window mode returns oldest -> newest order.  Reservoir mode returns
        insertion order of current reservoir slots (callers that need an
        unordered sample should shuffle).  Static mode (both sizes = 0)
        returns an empty list.
        """
        if self.refresh_window_size > 0:
            return list(self._accepted_deque)
        if self.reservoir_size > 0:
            return list(self._reservoir)
        return []

    def apply_refresh(
        self,
        *,
        scoring_model: nn.Module,
        mean: torch.Tensor,
        cov: torch.Tensor,
        scores: torch.Tensor,
        mean2: Optional[torch.Tensor] = None,
        cov2: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Replace scoring model, reference, and threshold atomically.

        Called by a ScoringRefresher after re-embedding bootstrap + accepted
        window through a fresh snapshot of the live training model.  The new
        threshold is recomputed at threshold_percentile so the static and
        adaptive filters share one calibration rule.

        Two-reference mode: when mean2 and cov2 are provided, they replace
        the secondary Gaussian and `scores` is expected to be the per-frame
        min(d_primary, d_secondary) over the union reference set, so the
        threshold percentile keeps the same intuitive meaning ("top-p of
        union distances are above threshold") under either mode.  Passing
        only one of mean2/cov2 is rejected.
        """
        threshold_before = self._threshold

        if (mean2 is None) ^ (cov2 is None):
            raise ValueError(
                "apply_refresh: mean2 and cov2 must be provided together "
                "(or both omitted for single-reference mode).",
            )

        self._scoring_model = scoring_model
        self.mean = mean.clone().float()
        self.cov = cov.clone().float()
        reg = 1e-5 * torch.eye(self.cov.shape[0])
        self._cov_inv = torch.linalg.inv(self.cov + reg)

        if mean2 is not None and cov2 is not None:
            self.mean2 = mean2.clone().float()
            self.cov2 = cov2.clone().float()
            reg2 = 1e-5 * torch.eye(self.cov2.shape[0])
            self._cov_inv2 = torch.linalg.inv(self.cov2 + reg2)
        else:
            self.mean2 = None
            self.cov2 = None
            self._cov_inv2 = None

        self._threshold = self._percentile(scores, self.threshold_percentile)

        self.num_refreshes += 1

        return {
            "num_refreshes": self.num_refreshes,
            "items_seen_at_refresh": self.items_seen,
            "window_size": len(self.get_accepted_frame_ids()),
            "threshold_before": float(threshold_before),
            "threshold_after": float(self._threshold),
            "reference_size": int(scores.numel()),
            "two_reference_active": self.mean2 is not None,
        }

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_accept + self.count_reject
        return {
            "count_accept": self.count_accept,
            "count_reject": self.count_reject,
            "accept_rate": self.count_accept / max(total, 1),
            "items_seen": self.items_seen,
            "accept_fraction": self.accept_fraction,
            "threshold_percentile": self.threshold_percentile,
            "current_threshold": self._threshold,
            "refresh_window_size": self.refresh_window_size,
            "reservoir_size": self.reservoir_size,
            "current_window_size": len(self.get_accepted_frame_ids()),
            "num_refreshes": self.num_refreshes,
        }


# =============================================================================
# DetectionUncertaintyPolicy
# =============================================================================


class DetectionUncertaintyPolicy(FilterPolicy):
    """
    Prediction-uncertainty filter with a bootstrap-calibrated threshold.

    For each new frame the frozen scoring model runs detection inference and
    a per-frame uncertainty score is derived from the top-K post-NMS
    classification confidences.  Frames with uncertainty at or above the
    threshold are accepted.  The threshold is calibrated from per-frame
    uncertainty scores over the reference set at threshold_percentile.

    Score definition:
        Two reductions are supported, selected by score_mode:

        topk_mean: score = 1 - mean(top_k box scores).  Low mean
        confidence on the most confident detections -> high uncertainty.

        margin: score = 1 - (top1 - top2 box scores).  Two near-equal
        top detections -> high uncertainty (ambiguous decision).  Falls
        back to 1 - top1 when the frame has a single detection.

        Both modes clamp to [0, 1].  Frames with zero detections score
        1.0 (maximum uncertainty -- the detector is effectively "blind"
        on that frame and is almost certainly informative to label).

    Why this signal:
        A Mahalanobis filter on backbone embeddings measures appearance
        novelty: how visually different a frame is from the bootstrap
        distribution.  Appearance novelty and task informativeness are
        correlated but not identical -- the detector may already handle a
        visually novel frame well (for example, a night scene with only
        the usual cars), and may still struggle with an appearance-familiar
        frame (for example, a daylight urban scene with many small
        pedestrians).  The uncertainty signal scores the latter directly.

    Adaptive refresh:
        An external ScoringRefresher can periodically re-score
        (bootstrap + accepted-frame reference) through a fresh snapshot of
        the live training model and call apply_refresh to replace the
        scoring model and the threshold.  The accepted-frame reference is
        populated in one of two mutually exclusive modes, matching
        DistributionBasedPolicy:

        refresh_window_size = M > 0: deque of the last M accepted frame
        ids (FIFO sliding window).

        reservoir_size = R > 0: uniform random sample of all past accepts
        maintained with Vitter's Algorithm R.

        With both sizes set to 0 the filter is static: no accepted frames
        participate in the reference, and a refresh only updates the
        scoring model and recomputes the threshold from refreshed
        bootstrap scores.

    Args:
        scoring_model: Frozen detector snapshot in eval mode with gradients
            disabled.  Called as scoring_model([image]) to obtain the
            list-of-prediction-dicts torchvision detection output; the
            "scores" field supplies the per-box confidences.
        bootstrap_scores: Per-frame uncertainty scores over the bootstrap
            frames, used to calibrate the initial threshold.
        threshold_percentile: Fraction of the reference distribution that
            should fall at or above the threshold.  E.g. 0.15 calibrates
            the threshold at the 85th percentile of bootstrap uncertainties;
            a stream frame is accepted iff its uncertainty is at least that
            large.
        accept_fraction: Nominal accept rate (logging only; the actual
            accept rate varies with stream content).
        top_k: Number of top-confidence detections to average when
            computing per-frame uncertainty in topk_mean mode.  Ignored
            by margin mode.  Defaults to 10.
        score_mode: Reduction used to turn per-box confidences into a
            per-frame score.  Either "topk_mean" (default) or "margin".
        refresh_window_size: If > 0, maintain a deque of the last M
            accepted frame ids.  Mutually exclusive with reservoir_size.
        reservoir_size: If > 0, maintain a uniform random reservoir of R
            accepted frame ids (Algorithm R).  Mutually exclusive with
            refresh_window_size.
        reservoir_seed: Seed for the reservoir's internal random.Random.
            Required when reservoir_size > 0 for reproducibility.
    """

    def __init__(
        self,
        *,
        scoring_model: nn.Module,
        bootstrap_scores: List[float],
        threshold_percentile: float = 0.15,
        accept_fraction: float = 0.15,
        top_k: int = 10,
        score_mode: Literal["topk_mean", "margin"] = "topk_mean",
        refresh_window_size: int = 0,
        reservoir_size: int = 0,
        reservoir_seed: Optional[int] = None,
    ):
        super().__init__()
        if scoring_model is None:
            raise ValueError(
                "DetectionUncertaintyPolicy requires a frozen scoring_model"
            )
        if not bootstrap_scores:
            raise ValueError(
                "DetectionUncertaintyPolicy requires bootstrap_scores to "
                "calibrate the threshold; run collect_uncertainties over "
                "the bootstrap frames."
            )
        if top_k <= 0:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if score_mode not in ("topk_mean", "margin"):
            raise ValueError(
                f"score_mode must be 'topk_mean' or 'margin', got {score_mode!r}"
            )

        refresh_window_size = max(0, int(refresh_window_size))
        reservoir_size = max(0, int(reservoir_size))
        if refresh_window_size > 0 and reservoir_size > 0:
            raise ValueError(
                "DetectionUncertaintyPolicy: refresh_window_size and "
                "reservoir_size are mutually exclusive; set at most one > 0."
            )

        self.accept_fraction = accept_fraction
        self.threshold_percentile = threshold_percentile
        self.top_k = int(top_k)
        self.score_mode: str = score_mode

        self._scoring_model = scoring_model
        self._threshold: float = self._percentile(
            bootstrap_scores, threshold_percentile,
        )

        self.items_seen = 0
        self.count_accept = 0
        self.count_reject = 0

        self.refresh_window_size: int = refresh_window_size
        self.reservoir_size: int = reservoir_size
        self._accepted_deque: deque[str] = deque(
            maxlen=self.refresh_window_size if self.refresh_window_size > 0 else 1,
        )
        self._reservoir: List[str] = []
        self._reservoir_rng: random.Random = random.Random(reservoir_seed)
        self.num_refreshes: int = 0

    @staticmethod
    def _percentile(
        scores: "List[float] | torch.Tensor", percentile: float,
    ) -> float:
        """Return the (1 - percentile) quantile of scores (top-p cutoff)."""
        if isinstance(scores, torch.Tensor):
            sorted_scores, _ = torch.sort(scores.float())
            arr = sorted_scores.tolist()
        else:
            arr = sorted(scores)
        if not arr:
            return 0.0
        idx = int(len(arr) * (1.0 - percentile))
        idx = min(max(idx, 0), len(arr) - 1)
        return float(arr[idx])

    def _compute_score(self, image: torch.Tensor) -> float:
        """Per-frame uncertainty score using the configured score_mode."""
        from ..training.streaming import _frame_uncertainty_score
        with torch.no_grad():
            preds = self._scoring_model([image])
        if not preds:
            return 1.0
        scores = preds[0].get("scores")
        return _frame_uncertainty_score(scores, self.top_k, self.score_mode)

    def _get_threshold(self) -> float:
        return self._threshold

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        device: torch.device,
    ) -> FilterResult:
        self.items_seen += 1

        image = stream_item.image.to(device)
        score = self._compute_score(image)
        threshold = self._threshold

        meta: Dict[str, Any] = {
            "metric": "uncertainty_score",
            "score": score,
            "threshold": threshold,
        }

        if score >= threshold:
            self.count_accept += 1
            self.selection_tracker.record("accept", stream_item.categories)
            self._record_accepted(stream_item)
            return ("accept", meta)

        self.count_reject += 1
        self.selection_tracker.record("reject", stream_item.categories)
        return ("reject", meta)

    def _record_accepted(self, stream_item: StreamItem) -> None:
        """Record an accepted frame-id in the active reference set (if any).

        Same dispatch as DistributionBasedPolicy._record_accepted: window
        mode appends to the FIFO deque; reservoir mode runs Vitter's
        Algorithm R over the accept count.  Frames without a frame_id in
        metadata are silently skipped.
        """
        frame_id = stream_item.metadata.get("frame_id")
        if frame_id is None:
            return
        fid = str(frame_id)

        if self.refresh_window_size > 0:
            self._accepted_deque.append(fid)
            return

        if self.reservoir_size > 0:
            n = self.count_accept  # 1-indexed Nth accept (post-increment in select_action).
            r = self.reservoir_size
            if n <= r:
                self._reservoir.append(fid)
            else:
                j = self._reservoir_rng.randrange(n)  # [0, n-1]
                if j < r:
                    self._reservoir[j] = fid

    def get_accepted_frame_ids(self) -> List[str]:
        """Return a snapshot of the current accepted-frame reference.

        Window mode returns oldest -> newest order.  Reservoir mode returns
        insertion order of current reservoir slots (callers that need an
        unordered sample should shuffle).  Static mode (both sizes = 0)
        returns an empty list.
        """
        if self.refresh_window_size > 0:
            return list(self._accepted_deque)
        if self.reservoir_size > 0:
            return list(self._reservoir)
        return []

    def apply_refresh(
        self,
        *,
        scoring_model: nn.Module,
        scores: torch.Tensor,
    ) -> Dict[str, Any]:
        """Replace scoring model and threshold atomically.

        Called by a ScoringRefresher after re-scoring (bootstrap + accepted
        reference) through a fresh snapshot of the live training model.
        The new threshold is recomputed at threshold_percentile so the
        static and adaptive uncertainty filters share one calibration rule
        (and match the rule used by DistributionBasedPolicy).
        """
        threshold_before = self._threshold

        self._scoring_model = scoring_model
        self._threshold = self._percentile(scores, self.threshold_percentile)

        self.num_refreshes += 1

        return {
            "num_refreshes": self.num_refreshes,
            "items_seen_at_refresh": self.items_seen,
            "window_size": len(self.get_accepted_frame_ids()),
            "threshold_before": float(threshold_before),
            "threshold_after": float(self._threshold),
            "reference_size": int(scores.numel()),
        }

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_accept + self.count_reject
        return {
            "count_accept": self.count_accept,
            "count_reject": self.count_reject,
            "accept_rate": self.count_accept / max(total, 1),
            "items_seen": self.items_seen,
            "accept_fraction": self.accept_fraction,
            "threshold_percentile": self.threshold_percentile,
            "current_threshold": self._threshold,
            "top_k": self.top_k,
            "score_mode": self.score_mode,
            "refresh_window_size": self.refresh_window_size,
            "reservoir_size": self.reservoir_size,
            "current_window_size": len(self.get_accepted_frame_ids()),
            "num_refreshes": self.num_refreshes,
        }


# =============================================================================
# MixturePolicy
# =============================================================================


class MixturePolicy(FilterPolicy):
    """
    Epsilon-greedy mixture of a signal-based inner policy and random.

    Per frame, draws u ~ U(0, 1) from an internal RNG:

        u < mixture_gamma: delegate to inner.select_action (signal path).
        u >= mixture_gamma: accept with probability accept_fraction
                            (random path, independent of content).

    Frames accepted via either path are recorded into the inner policy's
    accepted-frame reference (sliding window or reservoir), so a later
    refresh sees the actual training distribution rather than the
    signal-selected subset.  Frames scored via the random path are not
    run through the inner scoring model, so the wall-clock cost is
    roughly gamma x inner + (1 - gamma) x negligible.

    Rationale: pure signal-based AL (Mahalanobis, top-K uncertainty,
    margin) systematically under-samples easy in-distribution frames,
    shifting the training distribution away from the evaluation
    distribution.  Mixing with a random fraction restores coverage of
    common modes while retaining the exploratory pull toward novel or
    uncertain frames.  The overall accept rate satisfies

        overall_rate approx gamma * signal_rate + (1 - gamma) * accept_fraction

    and stays close to accept_fraction when the inner policy is
    calibrated at threshold_percentile = accept_fraction.

    Args:
        inner: Signal-based policy to delegate signal-path decisions to.
            Typically a DistributionBasedPolicy or DetectionUncertaintyPolicy.
            The inner policy owns the accepted-frame reference (sliding
            window or reservoir) and the refresh machinery.
        mixture_gamma: Fraction of frames routed to the inner policy.
            0.0 is equivalent to RandomPolicy(accept_fraction); 1.0 is
            equivalent to the inner policy alone.
        accept_fraction: Acceptance probability used on the random path.
            Also reported as the policy-level accept_fraction for
            logging symmetry with RandomPolicy.
        rng_seed: Seed for the internal random.Random governing routing
            and the random-path coin flip.  Independent of the inner
            policy's reservoir RNG.
    """

    def __init__(
        self,
        *,
        inner: FilterPolicy,
        mixture_gamma: float,
        accept_fraction: float,
        rng_seed: Optional[int] = None,
    ):
        super().__init__()
        if not 0.0 <= mixture_gamma <= 1.0:
            raise ValueError(
                f"mixture_gamma must be in [0, 1], got {mixture_gamma}"
            )
        if not 0.0 <= accept_fraction <= 1.0:
            raise ValueError(
                f"accept_fraction must be in [0, 1], got {accept_fraction}"
            )

        self.inner = inner
        self.mixture_gamma = float(mixture_gamma)
        self.accept_fraction = float(accept_fraction)
        self._rng = random.Random(rng_seed)

        self.items_seen = 0
        self.count_accept = 0
        self.count_reject = 0
        self.count_accept_signal = 0
        self.count_accept_random = 0
        self.count_signal_path = 0
        self.count_random_path = 0

    def select_action(
        self,
        stream_item: StreamItem,
        model: nn.Module,
        device: torch.device,
    ) -> FilterResult:
        self.items_seen += 1
        u = self._rng.random()

        if u < self.mixture_gamma:
            self.count_signal_path += 1
            action, inner_meta = self.inner.select_action(
                stream_item, model, device,
            )
            meta: Dict[str, Any] = dict(inner_meta)
            meta["path"] = "signal"
            meta["mixture_u"] = u
            if action == "accept":
                self.count_accept += 1
                self.count_accept_signal += 1
                self.selection_tracker.record("accept", stream_item.categories)
            else:
                self.count_reject += 1
                self.selection_tracker.record("reject", stream_item.categories)
            return action, meta

        self.count_random_path += 1
        r = self._rng.random()
        meta = {"path": "random", "mixture_u": u, "random_score": r}
        if r < self.accept_fraction:
            self.count_accept += 1
            self.count_accept_random += 1
            self.selection_tracker.record("accept", stream_item.categories)
            self._record_into_inner(stream_item)
            return ("accept", meta)

        self.count_reject += 1
        self.selection_tracker.record("reject", stream_item.categories)
        return ("reject", meta)

    def _record_into_inner(self, stream_item: StreamItem) -> None:
        """Route random-path accepts into the inner policy's reference.

        The inner policy's _record_accepted uses inner.count_accept as
        Vitter's 1-indexed N, so we increment it before delegating.
        Policies without an accepted-frame reference (NoFilterPolicy,
        RandomPolicy) have no _record_accepted hook and are silently
        skipped.
        """
        record_fn = getattr(self.inner, "_record_accepted", None)
        if record_fn is None:
            return
        # `count_accept` only exists on signal-based inner policies (those
        # with a reservoir/window).  Use getattr/setattr so the type
        # checker doesn't see this as an attribute access on the abstract
        # FilterPolicy base.
        cur = getattr(self.inner, "count_accept", None)
        if cur is not None:
            setattr(self.inner, "count_accept", cur + 1)
        record_fn(stream_item)

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_accept + self.count_reject
        stats: Dict[str, Any] = {
            "count_accept": self.count_accept,
            "count_reject": self.count_reject,
            "accept_rate": self.count_accept / max(total, 1),
            "items_seen": self.items_seen,
            "accept_fraction": self.accept_fraction,
            "mixture_gamma": self.mixture_gamma,
            "count_signal_path": self.count_signal_path,
            "count_random_path": self.count_random_path,
            "count_accept_signal": self.count_accept_signal,
            "count_accept_random": self.count_accept_random,
        }
        for k, v in self.inner.get_stats().items():
            stats[f"inner_{k}"] = v
        return stats

    def requires_model_forward(self) -> bool:
        return self.inner.requires_model_forward()
