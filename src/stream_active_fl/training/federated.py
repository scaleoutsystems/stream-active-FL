"""
Federated aggregation strategies.

Implements server-side model aggregation for simulated federated learning.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

import torch


def fedavg(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> OrderedDict[str, torch.Tensor]:
    """
    Federated Averaging (FedAvg) of model state dicts.

    Computes a weighted average of all parameters across clients.
    Frozen parameters (e.g. backbone) are identical across clients
    so averaging them is a harmless identity operation.

    Args:
        state_dicts: List of model.state_dict() from each client.
            Must all share the same keys and tensor shapes.
        weights: Optional per-client weights (e.g. number of training
            items).  If None, all clients are weighted equally.

    Returns:
        Averaged state dict ready for model.load_state_dict().

    Raises:
        ValueError: If state_dicts is empty.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")

    if len(state_dicts) == 1:
        return OrderedDict(
            (k, v.clone()) for k, v in state_dicts[0].items()
        )

    # Normalize weights
    if weights is None:
        n = len(state_dicts)
        norm_weights = [1.0 / n] * n
    else:
        total = sum(weights)
        if total == 0:
            n = len(state_dicts)
            norm_weights = [1.0 / n] * n
        else:
            norm_weights = [w / total for w in weights]

    # Weighted average via in-place accumulation (avoids N intermediate tensors
    # and the implicit int(0) start value of Python's built-in sum).
    avg_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in state_dicts[0]:
        acc = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
        for w, sd in zip(norm_weights, state_dicts):
            acc.add_(sd[key].float(), alpha=w)
        avg_state[key] = acc.to(state_dicts[0][key].dtype)

    return avg_state
