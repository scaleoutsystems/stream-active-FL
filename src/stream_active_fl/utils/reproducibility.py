"""Reproducibility utilities."""

from __future__ import annotations

import random
import warnings

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy, PyTorch CPU and CUDA,
    and configures cuDNN for deterministic behavior.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker init function that suppresses warnings.

    Use as worker_init_fn parameter in DataLoader to keep tqdm output clean.

    Args:
        worker_id: Worker ID (unused, required by DataLoader interface).
    """
    warnings.filterwarnings("ignore")
