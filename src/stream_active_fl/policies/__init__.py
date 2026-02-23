"""
Decision policies for selective training in stream learning.

Policies determine which stream items should trigger parameter updates (train),
be stored for replay (store), or be skipped entirely (skip).

Available policies:
    NoFilterPolicy          Train on every item (baseline)
    DifficultyBasedPolicy   Train only on high-loss items (adaptive or fixed threshold)
    TopKPolicy              Train on top-K hardest items in sliding window
    GradientNormPolicy      Train on items with largest parameter gradient norms
    TeacherConfidenceGate   Wrapper that filters out uncertain positive pseudo-labels

Factory:
    create_filter_policy    Build a FilterPolicy from an experiment config
"""

from __future__ import annotations

from typing import Any

from .filtering import (
    Action,
    FilterPolicy,
    FilterResult,
    NoFilterPolicy,
    DifficultyBasedPolicy,
    TopKPolicy,
    GradientNormPolicy,
    TeacherConfidenceGate,
)


def create_filter_policy(config: Any) -> FilterPolicy:
    """
    Create a filter policy from an experiment config dataclass.

    Reads filter_policy and policy-specific parameters from config,
    optionally wrapped with a TeacherConfidenceGate.

    Args:
        config: Any experiment config dataclass that exposes the standard
            filter-related fields (filter_policy, adaptive, train_fraction,
            loss_window_size, warmup_items, tau_loss, store_skipped,
            topk_window_size, topk_k, grad_norm_window_size, tau_grad_norm,
            tau_teacher, store_gated).

    Returns:
        Configured FilterPolicy instance.
    """
    if config.filter_policy == "none":
        policy: FilterPolicy = NoFilterPolicy()
    elif config.filter_policy == "difficulty":
        policy = DifficultyBasedPolicy(
            adaptive=config.adaptive,
            train_fraction=config.train_fraction,
            loss_window_size=config.loss_window_size,
            warmup_items=config.warmup_items,
            tau_loss=config.tau_loss,
            store_skipped=config.store_skipped,
        )
    elif config.filter_policy == "topk":
        policy = TopKPolicy(
            window_size=config.topk_window_size,
            k=config.topk_k,
        )
    elif config.filter_policy == "gradient_norm":
        policy = GradientNormPolicy(
            adaptive=config.adaptive,
            train_fraction=config.train_fraction,
            norm_window_size=config.grad_norm_window_size,
            warmup_items=config.warmup_items,
            tau_norm=config.tau_grad_norm,
            store_skipped=config.store_skipped,
        )
    else:
        raise ValueError(f"Unknown filter policy: {config.filter_policy}")

    if config.tau_teacher > 0.0:
        policy = TeacherConfidenceGate(
            inner_policy=policy,
            tau_teacher=config.tau_teacher,
            store_gated=config.store_gated,
        )

    return policy


__all__ = [
    "Action",
    "FilterPolicy",
    "FilterResult",
    "NoFilterPolicy",
    "DifficultyBasedPolicy",
    "TopKPolicy",
    "GradientNormPolicy",
    "TeacherConfidenceGate",
    "create_filter_policy",
]
