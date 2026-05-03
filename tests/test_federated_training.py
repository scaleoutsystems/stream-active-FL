"""Tests for stream_active_fl.training.federated (FedAvg)."""

from __future__ import annotations

import pytest
import torch

from stream_active_fl.training.federated import fedavg


def _make_state(val: float) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.full((4,), val),
        "bias": torch.full((2,), val * 2),
    }


def test_single_client_clones():
    sd = _make_state(3.0)
    result = fedavg([sd])
    torch.testing.assert_close(result["weight"], sd["weight"])
    assert result["weight"].data_ptr() != sd["weight"].data_ptr()


def test_equal_weight_average():
    sd1 = _make_state(1.0)
    sd2 = _make_state(3.0)
    result = fedavg([sd1, sd2])

    expected_w = torch.full((4,), 2.0)
    expected_b = torch.full((2,), 4.0)
    torch.testing.assert_close(result["weight"], expected_w)
    torch.testing.assert_close(result["bias"], expected_b)


def test_weighted_average():
    sd1 = _make_state(0.0)
    sd2 = _make_state(10.0)
    result = fedavg([sd1, sd2], weights=[1.0, 3.0])

    expected_w = torch.full((4,), 7.5)  # 0.25*0 + 0.75*10
    torch.testing.assert_close(result["weight"], expected_w)


def test_zero_weights_fallback_to_equal():
    sd1 = _make_state(2.0)
    sd2 = _make_state(4.0)
    result = fedavg([sd1, sd2], weights=[0.0, 0.0])

    expected = torch.full((4,), 3.0)
    torch.testing.assert_close(result["weight"], expected)


def test_empty_state_dicts_raises():
    with pytest.raises(ValueError, match="non-empty"):
        fedavg([])


def test_preserves_dtype():
    sd = {"p": torch.ones(3, dtype=torch.float16)}
    result = fedavg([sd, sd])
    assert result["p"].dtype == torch.float16
