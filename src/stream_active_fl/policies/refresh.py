"""Scoring-model refresh mechanism for the adaptive filters.

The ScoringRefresher periodically re-scores a fixed set of bootstrap
frames plus a sliding window or reservoir of the most recently accepted
stream frames using a fresh snapshot of the live training model.  The
refreshed reference replaces the static reference on the filter policy,
so the definition of "novel" or "hard" follows the model and the recent
accept history.

Typical cadence:
    Streaming: every K buffer flushes (configured at experiment level).
    Federated: every round, on the post-aggregation global model.

Supported policies:
    DistributionBasedPolicy -- Mahalanobis distance on backbone
    embeddings.  Refresh re-embeds the reference and updates
    (mean, cov, threshold).

    DetectionUncertaintyPolicy -- top-K detection-confidence
    uncertainty.  Refresh re-scores the reference and updates
    (scoring_model, threshold).

A single refresh event applies one snapshot and one reference pass to
the full list of policies passed in, so federated and streaming share
one server-side definition of novelty / hardness.

Design notes:
    Storage: only integer indices and frame-id strings are kept between
    refreshes -- no images, tensors, or scores are cached.  Frames are
    re-read from disk via a standard DataLoader at refresh time.

    Compute: each refresh costs one forward pass over
    (|bootstrap| + M) frames through the scoring model.  Embedding mode
    uses the backbone only; uncertainty mode runs the full detection
    head.  Both take ~20-90 s with AMP on a single GPU.

    Threshold: recomputed at the same threshold_percentile used at
    bootstrap calibration, so the static and adaptive filters share one
    calibration rule.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core.datasets import DetectionDataset, detection_collate
from ..utils import worker_init_fn
from .filtering import DetectionUncertaintyPolicy, DistributionBasedPolicy

ReferenceMode = Literal["single", "two_reference"]

RefreshablePolicy = Union[DistributionBasedPolicy, DetectionUncertaintyPolicy]


class _AcceptedFramesPolicy(Protocol):
    """Structural type used by `pool_recent_accepted`.

    Any policy that exposes its accepted frame-ids qualifies.  Reservoir
    mode is detected via an optional `reservoir_size` attribute (treated
    as 0 when absent).
    """

    def get_accepted_frame_ids(self) -> Sequence[str]: ...


@dataclass
class RefreshRecord:
    """Summary of a single refresh event (one row in refreshes.csv)."""

    refresh_idx: int
    trigger: str                # "buffer_flush" | "federated_round"
    trigger_count: int          # flush number or round number
    items_seen: int
    window_size: int
    reference_size: int
    threshold_before: float
    threshold_after: float
    duration_seconds: float


class ScoringRefresher:
    """Refresh the scoring model and the reference distribution.

    The refresher owns read-only metadata (manifest path, bootstrap
    frame entries, frame-id lookup, dataloader hyperparameters).  It is
    side-effect-free until .refresh(...) is called, which mutates the
    given policy in place and returns a RefreshRecord.
    """

    def __init__(
        self,
        *,
        manifest_path: Path,
        bootstrap_frame_entries: List[Dict[str, Any]],
        frame_id_to_entry: Dict[str, Dict[str, Any]],
        transform: Any,
        target_classes: Optional[List[str]],
        min_box_area: float,
        batch_size: int,
        num_workers: int,
        device: torch.device,
        use_amp: bool = True,
        reference_mode: ReferenceMode = "single",
        two_reference_min_accepts: int = 32,
        include_bootstrap: bool = True,
        no_bootstrap_min_accepts: int = 32,
    ):
        self.manifest_path = manifest_path
        self.bootstrap_frame_entries: List[Dict[str, Any]] = list(bootstrap_frame_entries)
        self.frame_id_to_entry = frame_id_to_entry
        self.transform = transform
        self.target_classes = target_classes
        self.min_box_area = min_box_area
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.use_amp = use_amp
        if reference_mode not in ("single", "two_reference"):
            raise ValueError(
                f"reference_mode must be 'single' or 'two_reference', got "
                f"{reference_mode!r}."
            )
        self.reference_mode: ReferenceMode = reference_mode
        # Below this many accepted frames, the secondary Gaussian's covariance
        # estimate is too noisy to be trusted; the refresher falls back to
        # single-reference for that refresh event so the filter still
        # behaves gracefully early in the stream.
        self.two_reference_min_accepts = max(1, int(two_reference_min_accepts))
        self.include_bootstrap = bool(include_bootstrap)
        # noBoot mode: when include_bootstrap=False, the bootstrap is
        # excluded from the reference at refresh time -- only the
        # window/reservoir of accepted frames is fitted.  The first refresh
        # event must have at least this many accepts for the covariance to
        # be stable; otherwise the refresh is a no-op (existing reference
        # is kept).  Two-reference mode is incompatible with noBoot (only
        # one source of frames -> degenerate to single Gaussian).
        if not self.include_bootstrap and self.reference_mode == "two_reference":
            raise ValueError(
                "include_bootstrap=False is incompatible with "
                "reference_mode='two_reference': there is only one set of "
                "frames (the accepted window/reservoir) so a second "
                "Gaussian cannot be fitted."
            )
        self.no_bootstrap_min_accepts = max(1, int(no_bootstrap_min_accepts))

        self._n_refreshes = 0

    @property
    def num_refreshes(self) -> int:
        return self._n_refreshes

    def _build_reference_entries(
        self, accepted_frame_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Assemble reference entries (bootstrap + accepted) deduplicated.

        When ``include_bootstrap`` is False, the bootstrap entries are
        skipped entirely and only the accepted window/reservoir is used.
        """
        if self.include_bootstrap:
            seen = {e["frame_id"] for e in self.bootstrap_frame_entries}
            entries = list(self.bootstrap_frame_entries)
        else:
            seen = set()
            entries = []
        for fid in accepted_frame_ids:
            if fid in seen:
                continue
            entry = self.frame_id_to_entry.get(fid)
            if entry is None:
                continue
            entries.append(entry)
            seen.add(fid)
        return entries

    def _make_loader(self, entries: List[Dict[str, Any]]) -> DataLoader:
        dataset = DetectionDataset(
            manifest_path=self.manifest_path,
            split="train",
            transform=self.transform,
            augmentation=None,
            min_box_area=self.min_box_area,
            target_classes=self.target_classes,
            verbose=False,
            frame_entries=entries,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=detection_collate,
            worker_init_fn=worker_init_fn,
            pin_memory=self.device.type == "cuda",
        )

    def _snapshot_scoring_model(self, live_model: nn.Module) -> nn.Module:
        """Return an eval-mode, no-grad deepcopy of the live model."""
        snapshot = copy.deepcopy(live_model)
        snapshot.eval()
        for p in snapshot.parameters():
            p.requires_grad = False
        snapshot.to(self.device)
        return snapshot

    def _refresh_distribution_reference(
        self,
        *,
        new_scoring: nn.Module,
        accepted_frame_ids: List[str],
    ) -> "Optional[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]":
        """Compute the new reference Gaussian(s) and threshold scores.

        Single-reference mode: fits one Gaussian to the union {bootstrap +
        accepted}.  Returns (mean, cov, count, scores, None, None) with
        scores being the per-frame Mahalanobis distances of the union to
        the union Gaussian (matches legacy bootstrap calibration).

        Two-reference mode: fits one Gaussian to the bootstrap re-embedded
        through the new scoring model and a second Gaussian to the
        accepted-frame window/reservoir.  Returns
        (mean_boot, cov_boot, count_union, min_scores, mean_adapt, cov_adapt)
        with min_scores = min(d_boot, d_adapt) per frame over the union.
        Falls back to single-reference for this refresh when fewer than
        ``two_reference_min_accepts`` accepted frames are available -- the
        secondary covariance estimate is too noisy below that threshold.

        noBoot single-reference mode (include_bootstrap=False): fits one
        Gaussian to the accepted-frame window/reservoir only; the bootstrap
        is excluded from both the reference and the threshold calibration.
        Returns None to signal "skip this refresh" if fewer than
        ``no_bootstrap_min_accepts`` accepts are available.
        """
        from ..training.streaming import collect_embeddings, mahalanobis_distances

        boot_entries = list(self.bootstrap_frame_entries)
        boot_ids = {e["frame_id"] for e in boot_entries}
        adapt_entries: List[Dict[str, Any]] = []
        for fid in accepted_frame_ids:
            if fid in boot_ids:
                continue
            entry = self.frame_id_to_entry.get(fid)
            if entry is not None:
                adapt_entries.append(entry)

        if not self.include_bootstrap:
            if len(adapt_entries) < self.no_bootstrap_min_accepts:
                return None
            loader = self._make_loader(adapt_entries)
            mean, cov, count, scores = collect_embeddings(
                new_scoring, loader, self.device,
                progress_bar=False, use_amp=self.use_amp,
            )
            return mean, cov, count, scores, None, None

        use_two_ref = (
            self.reference_mode == "two_reference"
            and len(adapt_entries) >= self.two_reference_min_accepts
        )

        if not use_two_ref:
            entries = boot_entries + adapt_entries
            loader = self._make_loader(entries)
            mean, cov, count, scores = collect_embeddings(
                new_scoring, loader, self.device,
                progress_bar=False, use_amp=self.use_amp,
            )
            return mean, cov, count, scores, None, None

        boot_loader = self._make_loader(boot_entries)
        adapt_loader = self._make_loader(adapt_entries)

        mean_boot, cov_boot, n_boot, _, emb_boot = collect_embeddings(
            new_scoring, boot_loader, self.device,
            progress_bar=False, use_amp=self.use_amp,
            return_embeddings=True,
        )
        mean_adapt, cov_adapt, n_adapt, _, emb_adapt = collect_embeddings(
            new_scoring, adapt_loader, self.device,
            progress_bar=False, use_amp=self.use_amp,
            return_embeddings=True,
        )

        union_emb = torch.cat([emb_boot, emb_adapt], dim=0)
        d_boot = mahalanobis_distances(union_emb, mean_boot, cov_boot)
        d_adapt = mahalanobis_distances(union_emb, mean_adapt, cov_adapt)
        min_scores = torch.minimum(d_boot, d_adapt)

        return (
            mean_boot, cov_boot, n_boot + n_adapt, min_scores,
            mean_adapt, cov_adapt,
        )

    def refresh(
        self,
        *,
        live_model: nn.Module,
        policies: Sequence[RefreshablePolicy],
        accepted_frame_ids: List[str],
        trigger: str,
        trigger_count: int,
    ) -> RefreshRecord:
        """Run one refresh against the given policies.

        A single scoring model + reference is computed once and applied
        to all policies in the list.  In the streaming experiment this
        is a list of length one; in federated it is the list of client
        policies (all share one reference, which is the natural reading
        of "one server, one fleet-wide definition of novelty").

        Dispatches on policy type:

        DistributionBasedPolicy: collect backbone embeddings and refresh
        (mean, cov, threshold).

        DetectionUncertaintyPolicy: collect per-frame detection-
        uncertainty scores and refresh (scoring_model, threshold).

        Mixed-type policy lists are rejected to keep the refresher
        invariant simple: one reference and one scoring signal per event.
        """
        from time import perf_counter

        if not policies:
            raise ValueError("At least one policy must be provided.")

        policy_types = {type(p) for p in policies}
        if len(policy_types) != 1:
            raise ValueError(
                "ScoringRefresher.refresh requires all policies to share a "
                f"type; got {[t.__name__ for t in policy_types]}."
            )
        policy_cls = next(iter(policy_types))

        start = perf_counter()
        threshold_before = policies[0]._get_threshold()
        items_seen = max(p.items_seen for p in policies)

        new_scoring = self._snapshot_scoring_model(live_model)

        if policy_cls is DistributionBasedPolicy:
            result = self._refresh_distribution_reference(
                new_scoring=new_scoring,
                accepted_frame_ids=accepted_frame_ids,
            )
            if result is None:
                # noBoot mode with too few accepts; keep existing reference
                # and threshold but still update the scoring snapshot so
                # the next refresh has the latest backbone embeddings.
                for policy in policies:
                    assert isinstance(policy, DistributionBasedPolicy)
                    policy._scoring_model = new_scoring
                reference_size = 0
            else:
                mean, cov, count, scores, mean2, cov2 = result
                for policy in policies:
                    assert isinstance(policy, DistributionBasedPolicy)
                    policy.apply_refresh(
                        scoring_model=new_scoring,
                        mean=mean,
                        cov=cov,
                        scores=scores,
                        mean2=mean2,
                        cov2=cov2,
                    )
                reference_size = count

        elif policy_cls is DetectionUncertaintyPolicy:
            from ..training.streaming import collect_uncertainties

            top_k_values = {p.top_k for p in policies}  # type: ignore[attr-defined]
            if len(top_k_values) != 1:
                raise ValueError(
                    "DetectionUncertaintyPolicy refresh requires all "
                    f"policies to share top_k; got {sorted(top_k_values)}."
                )
            top_k = next(iter(top_k_values))

            score_modes = {p.score_mode for p in policies}  # type: ignore[attr-defined]
            if len(score_modes) != 1:
                raise ValueError(
                    "DetectionUncertaintyPolicy refresh requires all "
                    f"policies to share score_mode; got {sorted(score_modes)}."
                )
            score_mode = next(iter(score_modes))

            entries = self._build_reference_entries(accepted_frame_ids)
            loader = self._make_loader(entries)
            scores = collect_uncertainties(
                new_scoring, loader, self.device,
                top_k=top_k, score_mode=score_mode, progress_bar=False,
            )

            for policy in policies:
                assert isinstance(policy, DetectionUncertaintyPolicy)
                policy.apply_refresh(
                    scoring_model=new_scoring,
                    scores=scores,
                )
            reference_size = int(scores.numel())

        else:
            raise TypeError(
                f"Unsupported policy type for refresh: {policy_cls.__name__}"
            )

        duration = perf_counter() - start
        threshold_after = policies[0]._get_threshold()
        self._n_refreshes += 1

        return RefreshRecord(
            refresh_idx=self._n_refreshes,
            trigger=trigger,
            trigger_count=trigger_count,
            items_seen=items_seen,
            window_size=len(accepted_frame_ids),
            reference_size=reference_size,
            threshold_before=float(threshold_before),
            threshold_after=float(threshold_after),
            duration_seconds=float(duration),
        )


def pool_recent_accepted(
    policies: Sequence[_AcceptedFramesPolicy],
    window_size: int,
    *,
    rng: "Optional[random.Random]" = None,
) -> List[str]:
    """Pool per-client accepted-frame buffers into a fleet-wide sample.

    Budget is split equally across policies (per-client quota).  The
    per-client sample is built mode-aware:

    Sliding-window mode (refresh_window_size > 0): take the client's
    most recent share accepts (tail of the deque).

    Reservoir mode (reservoir_size > 0): shuffle a copy of the reservoir
    with rng then take the first share entries.  Shuffling is required
    because the reservoir's internal order tracks when slots were last
    replaced, not the original accept order, so a plain tail would bias
    toward the most recently replaced slots.  An rng must be provided
    for any reservoir-mode policy.

    Static mode (both sizes = 0): the policy returns an empty list and
    contributes nothing.

    Duplicates are removed (first occurrence wins).  If some clients
    have fewer accepts than their share, the result is shorter than
    window_size; unused budget is not redistributed (keeping the pool
    balanced across clients is more important than maximizing reference
    size).
    """
    if window_size <= 0 or not policies:
        return []

    n = len(policies)
    base_share, remainder = divmod(window_size, n)

    result: List[str] = []
    seen: set[str] = set()
    for i, policy in enumerate(policies):
        share = base_share + (1 if i < remainder else 0)
        if share <= 0:
            continue
        fids = policy.get_accepted_frame_ids()
        if getattr(policy, "reservoir_size", 0) > 0:
            if rng is None:
                raise ValueError(
                    "pool_recent_accepted requires an rng when any client "
                    "policy is in reservoir mode"
                )
            fids = list(fids)
            rng.shuffle(fids)
            sample = fids[:share]
        else:
            sample = fids[-share:]
        for fid in sample:
            if fid in seen:
                continue
            seen.add(fid)
            result.append(fid)
    return result
