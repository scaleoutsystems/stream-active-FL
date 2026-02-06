"""
Decision policies for selective training in stream learning.

Policies determine which stream items should trigger parameter updates (train),
be stored for replay (store), or be skipped entirely (skip).

Available policies:
- NoFilterPolicy: Train on every item (baseline)
- DifficultyBasedPolicy: Train only on high-loss items with confident supervision
- TopKPolicy: Train on top-K hardest items in sliding window
"""

from .filtering import (
    Action,
    FilterPolicy,
    NoFilterPolicy,
    DifficultyBasedPolicy,
    TopKPolicy,
)

__all__ = [
    "Action",
    "FilterPolicy",
    "NoFilterPolicy",
    "DifficultyBasedPolicy",
    "TopKPolicy",
]
