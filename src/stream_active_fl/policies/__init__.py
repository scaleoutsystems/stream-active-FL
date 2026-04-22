"""Filter policies for streaming and federated experiments.

Public surface:

    NoFilterPolicy             Accept everything (baseline).
    RandomPolicy               Accept with fixed probability (baseline).
    DistributionBasedPolicy    Mahalanobis on backbone embeddings, optionally
                               refreshed via a sliding window or reservoir.
    DetectionUncertaintyPolicy Top-K mean or top1-top2 margin on post-NMS
                               confidences, optionally refreshed.
    MixturePolicy              Epsilon-greedy mixture of a signal-based inner
                               policy and random acceptance.

    ScoringRefresher           Periodic re-scoring of the reference under a
                               fresh snapshot of the live training model.
    pool_recent_accepted       Pool per-client reservoirs into a fleet-wide
                               reference (federated).

    create_filter_policy       Build a FilterPolicy from an experiment config.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from .filtering import (
    Action,
    DetectionUncertaintyPolicy,
    DistributionBasedPolicy,
    FilterPolicy,
    FilterResult,
    MixturePolicy,
    NoFilterPolicy,
    RandomPolicy,
)
from .refresh import RefreshRecord, ScoringRefresher, pool_recent_accepted


def create_filter_policy(
    config: Any,
    bootstrap_mean: Optional[torch.Tensor] = None,
    bootstrap_cov: Optional[torch.Tensor] = None,
    scoring_model: Optional[torch.nn.Module] = None,
    bootstrap_scores: Optional[List[float]] = None,
    reservoir_seed_override: Optional[int] = None,
) -> FilterPolicy:
    """Create a filter policy from an experiment config dataclass.

    Args:
        config: Experiment config with a filter_policy field and
            policy-specific parameters.  Supported filter_policy values:
            "none", "random", "distribution", "uncertainty",
            "mixed_distribution", "mixed_uncertainty".
        bootstrap_mean: Mean embedding from bootstrap phase.  Required
            for distribution-based policies (plain or mixed).
        bootstrap_cov: Covariance matrix from bootstrap phase.  Required
            for distribution-based policies (plain or mixed).
        scoring_model: Frozen model snapshot for per-frame scoring.
            Required for every signal-based policy (plain or mixed) so
            the score lives in a stable model state.
        bootstrap_scores: Per-frame reference scores for threshold
            calibration.  Mahalanobis distances for distribution
            policies; per-frame uncertainty scores for uncertainty
            policies.  The caller is responsible for producing them in
            the matching space (see collect_embeddings and
            collect_uncertainties).
        reservoir_seed_override: If given, overrides config.seed as the
            reservoir-sampler and mixture-routing seed.  Federated
            callers pass a client-unique value so per-client reservoirs
            and mixture draws are independent.

    Returns:
        Configured FilterPolicy instance.
    """
    policy = config.filter_policy

    if policy == "none":
        return NoFilterPolicy()

    if policy == "random":
        return RandomPolicy(accept_fraction=config.accept_fraction)

    def _build_distribution() -> DistributionBasedPolicy:
        if bootstrap_mean is None or bootstrap_cov is None:
            raise ValueError(
                "Distribution-based policy requires bootstrap_mean and "
                "bootstrap_cov.  Run bootstrap training first."
            )
        if scoring_model is None:
            raise ValueError(
                "Distribution-based policy requires a frozen scoring_model "
                "(pass a deepcopy of the bootstrap model)."
            )
        if not bootstrap_scores:
            raise ValueError(
                "Distribution-based policy requires bootstrap_scores to "
                "calibrate the threshold."
            )
        return DistributionBasedPolicy(
            bootstrap_mean=bootstrap_mean,
            bootstrap_cov=bootstrap_cov,
            scoring_model=scoring_model,
            bootstrap_scores=bootstrap_scores,
            accept_fraction=config.accept_fraction,
            threshold_percentile=getattr(config, "threshold_percentile", 0.10),
            refresh_window_size=getattr(config, "scoring_refresh_window_size", 0),
            reservoir_size=getattr(config, "scoring_refresh_reservoir_size", 0),
            reservoir_seed=(
                reservoir_seed_override
                if reservoir_seed_override is not None
                else getattr(config, "seed", None)
            ),
        )

    def _build_uncertainty() -> DetectionUncertaintyPolicy:
        if scoring_model is None:
            raise ValueError(
                "DetectionUncertaintyPolicy requires a frozen scoring_model "
                "(pass a deepcopy of the bootstrap model)."
            )
        if not bootstrap_scores:
            raise ValueError(
                "DetectionUncertaintyPolicy requires bootstrap_scores "
                "(per-frame detector uncertainty scores) to calibrate the "
                "threshold.  Run collect_uncertainties over the bootstrap "
                "frames before constructing the policy."
            )
        return DetectionUncertaintyPolicy(
            scoring_model=scoring_model,
            bootstrap_scores=bootstrap_scores,
            threshold_percentile=getattr(config, "threshold_percentile", 0.15),
            accept_fraction=config.accept_fraction,
            top_k=getattr(config, "uncertainty_top_k", 10),
            score_mode=getattr(config, "uncertainty_score_mode", "topk_mean"),
            refresh_window_size=getattr(config, "scoring_refresh_window_size", 0),
            reservoir_size=getattr(config, "scoring_refresh_reservoir_size", 0),
            reservoir_seed=(
                reservoir_seed_override
                if reservoir_seed_override is not None
                else getattr(config, "seed", None)
            ),
        )

    if policy == "distribution":
        return _build_distribution()

    if policy == "uncertainty":
        return _build_uncertainty()

    if policy in ("mixed_distribution", "mixed_uncertainty"):
        inner: FilterPolicy = (
            _build_distribution()
            if policy == "mixed_distribution"
            else _build_uncertainty()
        )
        return MixturePolicy(
            inner=inner,
            mixture_gamma=getattr(config, "mixture_gamma", 0.5),
            accept_fraction=config.accept_fraction,
            rng_seed=(
                reservoir_seed_override
                if reservoir_seed_override is not None
                else getattr(config, "seed", None)
            ),
        )

    raise ValueError(f"Unknown filter policy: {policy}")


__all__ = [
    "Action",
    "DetectionUncertaintyPolicy",
    "DistributionBasedPolicy",
    "FilterPolicy",
    "FilterResult",
    "MixturePolicy",
    "NoFilterPolicy",
    "RandomPolicy",
    "RefreshRecord",
    "ScoringRefresher",
    "create_filter_policy",
    "pool_recent_accepted",
]
