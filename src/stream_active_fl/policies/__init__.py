"""
Decision policies for selective training in buffer-based stream learning.

Policies determine which stream items should be accepted (added to the
training buffer) or rejected (discarded).

Available policies:
    NoFilterPolicy             Accept every item (baseline)
    RandomPolicy               Accept each item with fixed probability (random baseline)
    DistributionBasedPolicy    Accept items on the tail of the embedding distribution
    UncertaintyBasedPolicy     Accept items with high prediction uncertainty
    GradientNormPolicy         Accept items with largest parameter gradient norms

Factory:
    create_filter_policy       Build a FilterPolicy from an experiment config
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .filtering import (
    Action,
    DistributionBasedPolicy,
    FilterPolicy,
    FilterResult,
    GradientNormPolicy,
    NoFilterPolicy,
    RandomPolicy,
    UncertaintyBasedPolicy,
)


def create_filter_policy(
    config: Any,
    bootstrap_mean: Optional[torch.Tensor] = None,
    bootstrap_cov: Optional[torch.Tensor] = None,
    bootstrap_count: int = 0,
) -> FilterPolicy:
    """
    Create a filter policy from an experiment config dataclass.

    Args:
        config: Experiment config with filter_policy field and
            policy-specific parameters.
        bootstrap_mean: Mean embedding from bootstrap phase (required
            for distribution-based policy).
        bootstrap_cov: Covariance matrix from bootstrap phase (optional,
            used by distribution-based policy in mahalanobis mode).
        bootstrap_count: Number of bootstrap samples used to compute
            bootstrap_mean/bootstrap_cov.

    Returns:
        Configured FilterPolicy instance.
    """
    if config.filter_policy == "none":
        return NoFilterPolicy()

    elif config.filter_policy == "random":
        return RandomPolicy(accept_fraction=config.accept_fraction)

    elif config.filter_policy == "distribution":
        if bootstrap_mean is None:
            raise ValueError(
                "Distribution-based policy requires bootstrap_mean. "
                "Run bootstrap training first."
            )
        return DistributionBasedPolicy(
            bootstrap_mean=bootstrap_mean,
            bootstrap_cov=bootstrap_cov,
            bootstrap_count=bootstrap_count,
            mode=config.distribution_mode,
            accept_fraction=config.accept_fraction,
            score_window_size=config.score_window_size,
            warmup_items=config.warmup_items,
            embedding_buffer_size=config.embedding_buffer_size,
            knn_k=config.knn_k,
            update_stats=config.update_distribution_stats,
        )

    elif config.filter_policy == "uncertainty":
        return UncertaintyBasedPolicy(
            accept_fraction=config.accept_fraction,
            score_window_size=config.score_window_size,
            warmup_items=config.warmup_items,
            confidence_threshold=config.confidence_threshold,
            top_k_detections=config.top_k_detections,
        )

    elif config.filter_policy == "gradient_norm":
        return GradientNormPolicy(
            accept_fraction=config.accept_fraction,
            norm_window_size=config.norm_window_size,
            warmup_items=config.warmup_items,
        )

    else:
        raise ValueError(f"Unknown filter policy: {config.filter_policy}")


__all__ = [
    "Action",
    "DistributionBasedPolicy",
    "FilterPolicy",
    "FilterResult",
    "GradientNormPolicy",
    "NoFilterPolicy",
    "RandomPolicy",
    "UncertaintyBasedPolicy",
    "create_filter_policy",
]
