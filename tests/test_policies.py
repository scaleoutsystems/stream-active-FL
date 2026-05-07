"""Tests for filter policies and the refresh-pool helper."""

from __future__ import annotations

from typing import List, Optional, Set

import pytest
import torch
import torch.nn as nn

from stream_active_fl.core.items import StreamItem
from stream_active_fl.policies import create_filter_policy
from stream_active_fl.policies.filtering import (
    DetectionUncertaintyPolicy,
    DistributionBasedPolicy,
    MixturePolicy,
    NoFilterPolicy,
    RandomPolicy,
)
from stream_active_fl.policies.refresh import pool_recent_accepted


def _make_item(
    categories: Optional[Set[str]] = None,
    frame_id: str = "test",
    image: Optional[torch.Tensor] = None,
) -> StreamItem:
    return StreamItem(
        image=image if image is not None else torch.rand(3, 32, 32),
        annotations={
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),
        },
        categories=categories or {"Vehicle"},
        metadata={"frame_id": frame_id},
    )


class _DummyModel(nn.Module):
    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# NoFilterPolicy
# ---------------------------------------------------------------------------


def test_no_filter_always_accepts():
    policy = NoFilterPolicy()
    model = _DummyModel()
    device = torch.device("cpu")

    for _ in range(10):
        action, meta = policy.select_action(_make_item(), model, device)
        assert action == "accept"

    assert policy.count == 10
    assert policy.requires_model_forward() is False


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------


def test_random_policy_respects_fraction():
    policy = RandomPolicy(accept_fraction=0.5)
    model = _DummyModel()
    device = torch.device("cpu")

    n = 1000
    accepts = 0
    for _ in range(n):
        action, meta = policy.select_action(_make_item(), model, device)
        assert action in ("accept", "reject")
        assert "random_score" in meta
        if action == "accept":
            accepts += 1

    rate = accepts / n
    assert 0.35 < rate < 0.65, f"Expected ~0.5, got {rate}"


def test_random_policy_zero_fraction_rejects_all():
    policy = RandomPolicy(accept_fraction=0.0)
    model = _DummyModel()
    device = torch.device("cpu")

    for _ in range(50):
        action, _ = policy.select_action(_make_item(), model, device)
        assert action == "reject"


def test_random_policy_one_fraction_accepts_all():
    policy = RandomPolicy(accept_fraction=1.0)
    model = _DummyModel()
    device = torch.device("cpu")

    for _ in range(50):
        action, _ = policy.select_action(_make_item(), model, device)
        assert action == "accept"


def test_random_policy_requires_no_forward():
    assert RandomPolicy().requires_model_forward() is False


def test_random_policy_stats():
    policy = RandomPolicy(accept_fraction=1.0)
    model = _DummyModel()
    device = torch.device("cpu")

    for _ in range(5):
        policy.select_action(_make_item(), model, device)

    stats = policy.get_stats()
    assert stats["count_accept"] == 5
    assert stats["count_reject"] == 0
    assert stats["accept_rate"] == 1.0


# ---------------------------------------------------------------------------
# SelectionTracker (via policy)
# ---------------------------------------------------------------------------


def test_selection_tracker_per_category():
    policy = NoFilterPolicy()
    model = _DummyModel()
    device = torch.device("cpu")

    policy.select_action(_make_item({"Vehicle"}), model, device)
    policy.select_action(_make_item({"Pedestrian"}), model, device)
    policy.select_action(_make_item({"Vehicle", "Pedestrian"}), model, device)

    stats = policy.get_selection_stats()
    assert stats["accept_count"] == 3
    assert stats["accept_by_category"]["Vehicle"] == 2
    assert stats["accept_by_category"]["Pedestrian"] == 2


def test_selection_tracker_reset():
    policy = NoFilterPolicy()
    model = _DummyModel()
    device = torch.device("cpu")

    policy.select_action(_make_item(), model, device)
    policy.reset_selection_stats()

    stats = policy.get_selection_stats()
    assert stats["accept_count"] == 0
    assert stats["reject_count"] == 0


# ---------------------------------------------------------------------------
# DistributionBasedPolicy
# ---------------------------------------------------------------------------


class _MagnitudeScorer(nn.Module):
    """Scorer whose embedding is driven by the image mean.

    get_embedding([image]) returns [image.mean().item(), 0, ..., 0] so
    tests can inject known Mahalanobis distances.
    """

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

    def get_embedding(self, images):
        vecs = []
        for img in images:
            v = torch.zeros(self.embed_dim)
            v[0] = float(img.mean().item())
            vecs.append(v)
        return torch.stack(vecs, dim=0)


def _image_with_mean(value: float, shape=(3, 8, 8)) -> torch.Tensor:
    return torch.full(shape, float(value))


def _make_dist_policy(
    *,
    bootstrap_scores: list[float],
    refresh_window_size: int = 0,
    threshold_percentile: float = 0.10,
) -> DistributionBasedPolicy:
    """Build a policy with identity covariance and zero mean (cpu)."""
    embed_dim = 4
    mean = torch.zeros(embed_dim)
    cov = torch.eye(embed_dim)
    return DistributionBasedPolicy(
        bootstrap_mean=mean,
        bootstrap_cov=cov,
        scoring_model=_MagnitudeScorer(embed_dim=embed_dim),
        bootstrap_scores=bootstrap_scores,
        threshold_percentile=threshold_percentile,
        refresh_window_size=refresh_window_size,
    )


def test_dist_policy_threshold_calibration():
    scores = [float(i) for i in range(100)]
    p10 = _make_dist_policy(bootstrap_scores=scores, threshold_percentile=0.10)
    assert p10._get_threshold() == 90.0

    p50 = _make_dist_policy(bootstrap_scores=scores, threshold_percentile=0.50)
    assert p50._get_threshold() == 50.0

    p100 = _make_dist_policy(bootstrap_scores=scores, threshold_percentile=1.0)
    assert p100._get_threshold() == 0.0


def test_dist_policy_accepts_scores_at_or_above_threshold():
    scores = [float(i) for i in range(100)]
    policy = _make_dist_policy(bootstrap_scores=scores, threshold_percentile=0.10)
    model = _DummyModel()
    device = torch.device("cpu")

    assert policy._get_threshold() == 90.0

    high = _make_item(image=_image_with_mean(95.0), frame_id="hi")
    action, meta = policy.select_action(high, model, device)
    assert action == "accept"
    assert abs(meta["score"] - 95.0) < 1e-3

    well_above = _make_item(image=_image_with_mean(91.0), frame_id="above")
    action, _ = policy.select_action(well_above, model, device)
    assert action == "accept"

    low = _make_item(image=_image_with_mean(10.0), frame_id="lo")
    action, _ = policy.select_action(low, model, device)
    assert action == "reject"

    stats = policy.get_stats()
    assert stats["count_accept"] == 2
    assert stats["count_reject"] == 1
    assert stats["items_seen"] == 3


def test_dist_policy_window_disabled_records_nothing():
    policy = _make_dist_policy(
        bootstrap_scores=[float(i) for i in range(100)],
        refresh_window_size=0,
    )
    model, device = _DummyModel(), torch.device("cpu")

    for i in range(5):
        item = _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}")
        policy.select_action(item, model, device)

    assert policy.count_accept == 5
    assert policy.get_accepted_frame_ids() == []


def test_dist_policy_window_enabled_respects_maxlen():
    policy = _make_dist_policy(
        bootstrap_scores=[float(i) for i in range(100)],
        refresh_window_size=3,
    )
    model, device = _DummyModel(), torch.device("cpu")

    for i in range(5):
        item = _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}")
        action, _ = policy.select_action(item, model, device)
        assert action == "accept"

    assert policy.get_accepted_frame_ids() == ["f2", "f3", "f4"]


def test_dist_policy_rejected_frames_not_recorded():
    policy = _make_dist_policy(
        bootstrap_scores=[float(i) for i in range(100)],
        refresh_window_size=5,
    )
    model, device = _DummyModel(), torch.device("cpu")

    policy.select_action(_make_item(image=_image_with_mean(95.0), frame_id="ok"),
                         model, device)
    policy.select_action(_make_item(image=_image_with_mean(10.0), frame_id="no"),
                         model, device)

    assert policy.get_accepted_frame_ids() == ["ok"]


def test_dist_policy_apply_refresh_updates_state_atomically():
    policy = _make_dist_policy(
        bootstrap_scores=[float(i) for i in range(100)],
        refresh_window_size=3,
        threshold_percentile=0.10,
    )
    prev_threshold = policy._get_threshold()
    prev_scorer = policy._scoring_model

    new_scorer = _MagnitudeScorer(embed_dim=4)
    new_mean = torch.ones(4)
    new_cov = 2.0 * torch.eye(4)
    new_scores = torch.tensor([float(i) for i in range(200)])

    info = policy.apply_refresh(
        scoring_model=new_scorer,
        mean=new_mean,
        cov=new_cov,
        scores=new_scores,
    )

    assert policy._scoring_model is new_scorer
    assert policy._scoring_model is not prev_scorer
    assert torch.equal(policy.mean, new_mean.float())
    assert torch.equal(policy.cov, new_cov.float())
    assert policy._get_threshold() == 180.0
    assert policy._get_threshold() != prev_threshold
    assert policy.num_refreshes == 1
    assert info["reference_size"] == 200
    assert info["threshold_before"] == prev_threshold
    assert info["threshold_after"] == 180.0

    probe = torch.tensor([3.0, 1.0, 1.0, 1.0])
    diff = probe - new_mean.float()
    expected_score = float(torch.sqrt(diff @ torch.linalg.inv(
        new_cov.float() + 1e-5 * torch.eye(4)) @ diff).item())
    got = policy._compute_score(probe)
    assert abs(got - expected_score) < 1e-5
    assert got > 0.0


# ---------------------------------------------------------------------------
# pool_recent_accepted
# ---------------------------------------------------------------------------


class _FakeDistPolicy:
    """Lightweight stand-in with the only API pool_recent_accepted reads."""

    def __init__(self, frame_ids: list[str]):
        self._ids = list(frame_ids)

    def get_accepted_frame_ids(self) -> list[str]:
        return list(self._ids)


def test_pool_empty_window_is_empty():
    policies = [_FakeDistPolicy([f"c0_{i}" for i in range(10)])]
    assert pool_recent_accepted(policies, window_size=0) == []
    assert pool_recent_accepted(policies, window_size=-5) == []


def test_pool_no_policies_is_empty():
    assert pool_recent_accepted([], window_size=100) == []


def test_pool_single_policy_returns_tail():
    p = _FakeDistPolicy([f"f{i}" for i in range(20)])
    pooled = pool_recent_accepted([p], window_size=5)
    assert pooled == ["f15", "f16", "f17", "f18", "f19"]


def test_pool_per_client_quota_balances_across_clients():
    """Each client contributes window_size // N of its tail.

    With 4 clients and window_size=100, each client contributes its
    last 25 accepts -- no single client dominates the pooled reference.
    """
    policies = [
        _FakeDistPolicy([f"c{c}_{i}" for i in range(100)]) for c in range(4)
    ]
    pooled = pool_recent_accepted(policies, window_size=100)

    assert len(pooled) == 100
    per_client: dict[str, int] = {}
    for fid in pooled:
        cid = fid.split("_")[0]
        per_client[cid] = per_client.get(cid, 0) + 1
    assert per_client == {"c0": 25, "c1": 25, "c2": 25, "c3": 25}

    assert pooled[:25] == [f"c0_{i}" for i in range(75, 100)]
    assert pooled[25:50] == [f"c1_{i}" for i in range(75, 100)]
    assert pooled[50:75] == [f"c2_{i}" for i in range(75, 100)]
    assert pooled[75:100] == [f"c3_{i}" for i in range(75, 100)]


def test_pool_uneven_deques_do_not_overfill():
    """Clients with fewer accepts than their share contribute all they have."""
    policies = [
        _FakeDistPolicy([f"big_{i}" for i in range(100)]),
        _FakeDistPolicy([f"mid_{i}" for i in range(10)]),
        _FakeDistPolicy([]),
        _FakeDistPolicy([f"sml_{i}" for i in range(3)]),
    ]
    pooled = pool_recent_accepted(policies, window_size=100)

    assert sum(1 for f in pooled if f.startswith("big_")) == 25
    assert sum(1 for f in pooled if f.startswith("mid_")) == 10
    assert sum(1 for f in pooled if f.startswith("sml_")) == 3
    assert len(pooled) == 25 + 10 + 3


def test_pool_uneven_window_distributes_remainder_to_first_clients():
    """window_size=10, 4 clients -> shares of 3, 3, 2, 2."""
    policies = [
        _FakeDistPolicy([f"c{c}_{i}" for i in range(20)]) for c in range(4)
    ]
    pooled = pool_recent_accepted(policies, window_size=10)

    per_client: dict[str, int] = {}
    for fid in pooled:
        cid = fid.split("_")[0]
        per_client[cid] = per_client.get(cid, 0) + 1
    assert per_client == {"c0": 3, "c1": 3, "c2": 2, "c3": 2}


def test_pool_deduplicates_shared_frame_ids():
    """If two clients accept the same frame-id, it appears once."""
    policies = [
        _FakeDistPolicy(["shared", "a1", "a2", "a3"]),
        _FakeDistPolicy(["shared", "b1", "b2", "b3"]),
    ]
    pooled = pool_recent_accepted(policies, window_size=8)
    assert pooled.count("shared") == 1
    assert set(pooled) == {"shared", "a1", "a2", "a3", "b1", "b2", "b3"}


def test_pool_takes_client_tail_not_head():
    """Regression: the pool must reflect each client's MOST recent accepts."""
    p = _FakeDistPolicy([f"f{i}" for i in range(10)])
    pooled = pool_recent_accepted([p], window_size=3)
    assert pooled == ["f7", "f8", "f9"]


# ---------------------------------------------------------------------------
# DistributionBasedPolicy: reservoir sampling (Algorithm R)
# ---------------------------------------------------------------------------


def _make_reservoir_policy(
    *,
    reservoir_size: int,
    reservoir_seed: int = 0,
    bootstrap_scores: Optional[List[float]] = None,
    threshold_percentile: float = 0.10,
) -> DistributionBasedPolicy:
    """Identity-covariance / zero-mean policy with reservoir mode enabled."""
    embed_dim = 4
    return DistributionBasedPolicy(
        bootstrap_mean=torch.zeros(embed_dim),
        bootstrap_cov=torch.eye(embed_dim),
        scoring_model=_MagnitudeScorer(embed_dim=embed_dim),
        bootstrap_scores=bootstrap_scores or [float(i) for i in range(100)],
        threshold_percentile=threshold_percentile,
        reservoir_size=reservoir_size,
        reservoir_seed=reservoir_seed,
    )


def test_dist_policy_window_and_reservoir_mutually_exclusive():
    import pytest

    with pytest.raises(ValueError, match="mutually exclusive"):
        DistributionBasedPolicy(
            bootstrap_mean=torch.zeros(4),
            bootstrap_cov=torch.eye(4),
            scoring_model=_MagnitudeScorer(embed_dim=4),
            bootstrap_scores=[1.0, 2.0, 3.0],
            refresh_window_size=5,
            reservoir_size=5,
        )


def test_reservoir_first_R_accepts_fill_directly():
    policy = _make_reservoir_policy(reservoir_size=3)
    model, device = _DummyModel(), torch.device("cpu")

    for i in range(3):
        item = _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}")
        policy.select_action(item, model, device)
    assert policy.get_accepted_frame_ids() == ["f0", "f1", "f2"]


def test_reservoir_size_saturates_at_R():
    policy = _make_reservoir_policy(reservoir_size=5)
    model, device = _DummyModel(), torch.device("cpu")

    for i in range(100):
        item = _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}")
        policy.select_action(item, model, device)
    assert len(policy.get_accepted_frame_ids()) == 5
    assert policy.count_accept == 100


def test_reservoir_rejected_frames_not_added():
    policy = _make_reservoir_policy(reservoir_size=5)
    model, device = _DummyModel(), torch.device("cpu")

    policy.select_action(_make_item(image=_image_with_mean(95.0), frame_id="ok"),
                         model, device)
    policy.select_action(_make_item(image=_image_with_mean(10.0), frame_id="no"),
                         model, device)
    assert policy.get_accepted_frame_ids() == ["ok"]


def test_reservoir_deterministic_with_same_seed():
    """Same seed + same accept sequence yields the same reservoir."""
    model, device = _DummyModel(), torch.device("cpu")
    results = []
    for _ in range(2):
        policy = _make_reservoir_policy(reservoir_size=10, reservoir_seed=123)
        for i in range(200):
            item = _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}")
            policy.select_action(item, model, device)
        results.append(tuple(policy.get_accepted_frame_ids()))
    assert results[0] == results[1]


def test_reservoir_ignores_global_random_seed():
    """Reservoir sampling must not read from the global random module."""
    import random as _r

    model, device = _DummyModel(), torch.device("cpu")
    outputs = []
    for outer_seed in (0, 12345):
        _r.seed(outer_seed)
        policy = _make_reservoir_policy(reservoir_size=10, reservoir_seed=42)
        for i in range(200):
            item = _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}")
            policy.select_action(item, model, device)
        outputs.append(tuple(policy.get_accepted_frame_ids()))
    assert outputs[0] == outputs[1]


def test_reservoir_uniform_sample_frequencies():
    """Algorithm R: each past accept appears with freq ~= R/N.

    We run many trials with independent seeds, N=200 accepts, R=20.
    Expected occurrence frequency for any single frame is 20/200 = 0.1.
    With 500 trials the Monte Carlo std on that mean is ~sqrt(0.1*0.9/500)
    ~ 0.013, so we allow a wide +/- 0.04 band.
    """
    model, device = _DummyModel(), torch.device("cpu")
    N, R, trials = 200, 20, 500
    counts = [0] * N
    for seed in range(trials):
        policy = _make_reservoir_policy(reservoir_size=R, reservoir_seed=seed)
        for i in range(N):
            item = _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}")
            policy.select_action(item, model, device)
        present = {fid for fid in policy.get_accepted_frame_ids()}
        for i in range(N):
            if f"f{i}" in present:
                counts[i] += 1

    expected = R / N
    frequencies = [c / trials for c in counts]
    worst_dev = max(abs(f - expected) for f in frequencies)
    assert worst_dev < 0.04, (
        f"Reservoir distribution not uniform: worst deviation {worst_dev} "
        f"(expected ~0, tolerance 0.04)"
    )


def test_reservoir_apply_refresh_preserves_reservoir():
    """A scoring-model refresh must not clear the reservoir contents."""
    policy = _make_reservoir_policy(reservoir_size=5)
    model, device = _DummyModel(), torch.device("cpu")

    for i in range(10):
        item = _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}")
        policy.select_action(item, model, device)
    snapshot = tuple(policy.get_accepted_frame_ids())
    assert len(snapshot) == 5

    policy.apply_refresh(
        scoring_model=_MagnitudeScorer(embed_dim=4),
        mean=torch.ones(4),
        cov=2.0 * torch.eye(4),
        scores=torch.tensor([float(i) for i in range(100)]),
    )
    assert tuple(policy.get_accepted_frame_ids()) == snapshot


def test_pool_reservoir_requires_rng():
    """Pooling from a reservoir-mode client without rng must raise."""
    import pytest

    policy = _make_reservoir_policy(reservoir_size=3)
    model, device = _DummyModel(), torch.device("cpu")
    for i in range(5):
        policy.select_action(
            _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}"),
            model, device,
        )
    with pytest.raises(ValueError, match="reservoir mode"):
        pool_recent_accepted([policy], window_size=3)


def test_pool_reservoir_with_rng_samples_from_reservoir():
    """With rng provided, pooling returns entries drawn from the reservoir."""
    import random as _r

    policy = _make_reservoir_policy(reservoir_size=4, reservoir_seed=7)
    model, device = _DummyModel(), torch.device("cpu")
    for i in range(50):
        policy.select_action(
            _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}"),
            model, device,
        )
    reservoir = set(policy.get_accepted_frame_ids())
    assert len(reservoir) == 4

    pooled = pool_recent_accepted([policy], window_size=3, rng=_r.Random(0))
    assert len(pooled) == 3
    assert set(pooled) <= reservoir


# ---------------------------------------------------------------------------
# DetectionUncertaintyPolicy
# ---------------------------------------------------------------------------


class _ConstantUncertaintyDetector(nn.Module):
    """Detector stub whose per-frame confidence is the image's mean.

    Calling the model on a list of images returns torchvision-style
    predictions: one dict per image with a "scores" tensor of length 5,
    all filled with the image's mean value.  Uncertainty is then
    1 - image.mean() (assuming image.mean() in [0, 1]).
    """

    def forward(self, images):
        out = []
        for img in images:
            mean = float(img.mean().item())
            out.append(
                {
                    "scores": torch.full((5,), mean, dtype=torch.float32),
                    "boxes": torch.zeros((5, 4)),
                    "labels": torch.zeros((5,), dtype=torch.int64),
                }
            )
        return out


def _unit_image(value: float, shape=(3, 4, 4)) -> torch.Tensor:
    return torch.full(shape, float(value))


def test_uncertainty_threshold_accepts_uncertain_frames():
    """score = 1 - top-K mean confidence; high uncertainty -> accept."""
    import pytest

    scoring = _ConstantUncertaintyDetector()
    bootstrap_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    policy = DetectionUncertaintyPolicy(
        scoring_model=scoring,
        bootstrap_scores=bootstrap_scores,
        threshold_percentile=0.3,
        top_k=5,
    )
    device = torch.device("cpu")

    # confident frame (mean=0.9) -> uncertainty 0.1 -> reject
    action, meta = policy.select_action(
        _make_item(image=_unit_image(0.9)), _DummyModel(), device,
    )
    assert action == "reject"
    assert meta["score"] == pytest.approx(0.1, abs=1e-6)

    # uncertain frame (mean=0.1) -> uncertainty 0.9 -> accept
    action, meta = policy.select_action(
        _make_item(image=_unit_image(0.1)), _DummyModel(), device,
    )
    assert action == "accept"
    assert meta["score"] == pytest.approx(0.9, abs=1e-6)


def test_uncertainty_no_detections_scores_one():
    """Frames the detector returns zero boxes for score 1.0 (max uncertain)."""
    import pytest as _pytest

    class _EmptyDetector(nn.Module):
        def forward(self, images):
            return [
                {
                    "scores": torch.empty(0, dtype=torch.float32),
                    "boxes": torch.empty((0, 4)),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }
                for _ in images
            ]

    policy = DetectionUncertaintyPolicy(
        scoring_model=_EmptyDetector(),
        bootstrap_scores=[0.5],
        threshold_percentile=0.5,
        top_k=5,
    )
    action, meta = policy.select_action(
        _make_item(image=_unit_image(0.5)), _DummyModel(), torch.device("cpu"),
    )
    assert action == "accept"
    assert meta["score"] == _pytest.approx(1.0, abs=1e-6)


def test_uncertainty_requires_bootstrap_scores():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="bootstrap_scores"):
        DetectionUncertaintyPolicy(
            scoring_model=_ConstantUncertaintyDetector(),
            bootstrap_scores=[],
        )


def test_uncertainty_reservoir_apply_refresh_preserves_reservoir():
    """Refresh on a reservoir-mode uncertainty policy must not clear it."""
    policy = DetectionUncertaintyPolicy(
        scoring_model=_ConstantUncertaintyDetector(),
        bootstrap_scores=[0.5],
        threshold_percentile=0.5,
        top_k=5,
        reservoir_size=5,
        reservoir_seed=123,
    )
    threshold_before = policy._threshold
    device = torch.device("cpu")

    # Drive 20 accepts with high-uncertainty frames (mean=0.1 -> unc=0.9).
    for i in range(20):
        action, _ = policy.select_action(
            _make_item(image=_unit_image(0.1), frame_id=f"f{i}"),
            _DummyModel(), device,
        )
        assert action == "accept"

    snapshot = tuple(policy.get_accepted_frame_ids())
    assert len(snapshot) == 5
    assert policy.count_accept == 20

    # Refresh with a fresh scorer and a clearly different reference distribution.
    new_scorer = _ConstantUncertaintyDetector()
    record = policy.apply_refresh(
        scoring_model=new_scorer,
        scores=torch.tensor([0.05, 0.10, 0.20, 0.80, 0.95]),
    )

    # Reservoir preserved; scorer and threshold replaced; counters bumped.
    assert tuple(policy.get_accepted_frame_ids()) == snapshot
    assert policy._scoring_model is new_scorer
    assert policy._threshold != threshold_before
    assert policy.num_refreshes == 1
    assert record["reference_size"] == 5
    assert record["window_size"] == 5


def test_uncertainty_window_apply_refresh_replaces_threshold():
    """Refresh on a window-mode uncertainty policy updates threshold/scorer."""
    import pytest

    policy = DetectionUncertaintyPolicy(
        scoring_model=_ConstantUncertaintyDetector(),
        bootstrap_scores=[0.5],
        threshold_percentile=0.5,
        top_k=5,
        refresh_window_size=3,
    )
    device = torch.device("cpu")
    for i in range(6):
        policy.select_action(
            _make_item(image=_unit_image(0.1), frame_id=f"f{i}"),
            _DummyModel(), device,
        )

    # Window keeps only the 3 most recent accepts.
    assert policy.get_accepted_frame_ids() == ["f3", "f4", "f5"]

    new_scorer = _ConstantUncertaintyDetector()
    record = policy.apply_refresh(
        scoring_model=new_scorer,
        scores=torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]),
    )
    assert policy._scoring_model is new_scorer
    # Percentile 0.5 of length-5 scores -> idx = int(5 * 0.5) = 2 -> 0.6.
    assert policy._threshold == pytest.approx(0.6, abs=1e-6)
    assert record["threshold_after"] == pytest.approx(0.6, abs=1e-6)


# ---------------------------------------------------------------------------
# DistributionBasedPolicy: two-reference (multimodal) scoring
# ---------------------------------------------------------------------------


def test_dist_policy_two_reference_score_takes_min():
    """`_compute_score` returns min(d_primary, d_secondary) when mean2/cov2 set."""
    policy = _make_dist_policy(bootstrap_scores=[1.0, 2.0, 3.0])
    policy.apply_refresh(
        scoring_model=_MagnitudeScorer(embed_dim=4),
        mean=torch.zeros(4),
        cov=torch.eye(4),
        scores=torch.tensor([1.0, 2.0, 3.0]),
        mean2=torch.tensor([5.0, 0.0, 0.0, 0.0]),
        cov2=torch.eye(4),
    )

    # Tolerance accommodates the 1e-5 covariance regularization eps used
    # in apply_refresh; without that, distances would match exactly.
    tol = 1e-3

    # Probe near the second mode -> distance to mode 2 ~ 0, to mode 1 ~ 5.
    near_mode2 = torch.tensor([5.0, 0.0, 0.0, 0.0])
    s2 = policy._compute_score(near_mode2)
    assert s2 == pytest.approx(0.0, abs=tol)

    # Probe near the origin -> distance to mode 1 ~ 0, to mode 2 ~ 5.
    near_mode1 = torch.zeros(4)
    s1 = policy._compute_score(near_mode1)
    assert s1 == pytest.approx(0.0, abs=tol)

    # Probe between modes -> ~2.5 from each, min stays ~2.5.
    between = torch.tensor([2.5, 0.0, 0.0, 0.0])
    sm = policy._compute_score(between)
    assert sm == pytest.approx(2.5, abs=tol)


def test_dist_policy_two_reference_apply_refresh_single_then_two_then_back():
    """Switching modes via apply_refresh leaves no stale secondary state."""
    policy = _make_dist_policy(bootstrap_scores=[float(i) for i in range(10)])
    assert policy.mean2 is None and policy._cov_inv2 is None

    # Promote to two-reference.
    policy.apply_refresh(
        scoring_model=_MagnitudeScorer(embed_dim=4),
        mean=torch.zeros(4),
        cov=torch.eye(4),
        scores=torch.tensor([1.0, 2.0, 3.0]),
        mean2=torch.tensor([10.0, 0.0, 0.0, 0.0]),
        cov2=torch.eye(4),
    )
    assert policy.mean2 is not None and policy._cov_inv2 is not None

    # Demote back to single-reference; mean2/cov2 must be cleared.
    policy.apply_refresh(
        scoring_model=_MagnitudeScorer(embed_dim=4),
        mean=torch.zeros(4),
        cov=torch.eye(4),
        scores=torch.tensor([1.0, 2.0, 3.0]),
    )
    assert policy.mean2 is None
    assert policy.cov2 is None
    assert policy._cov_inv2 is None


def test_dist_policy_two_reference_partial_args_rejected():
    """Passing only one of mean2/cov2 must raise (atomic two-ref state)."""
    policy = _make_dist_policy(bootstrap_scores=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="must be provided together"):
        policy.apply_refresh(
            scoring_model=_MagnitudeScorer(embed_dim=4),
            mean=torch.zeros(4),
            cov=torch.eye(4),
            scores=torch.tensor([1.0]),
            mean2=torch.ones(4),
            cov2=None,  # only one provided
        )


# ---------------------------------------------------------------------------
# MixturePolicy
# ---------------------------------------------------------------------------


class _AlwaysAcceptInner(NoFilterPolicy):
    """Stand-in inner policy that records into a list and always accepts."""

    def __init__(self):
        super().__init__()
        self.recorded: list[str] = []
        self.count_accept = 0  # exposed for MixturePolicy._record_into_inner

    def _record_accepted(self, item: StreamItem) -> None:
        self.recorded.append(item.metadata["frame_id"])


def test_mixture_policy_validates_arguments():
    inner = NoFilterPolicy()
    with pytest.raises(ValueError, match="mixture_gamma"):
        MixturePolicy(inner=inner, mixture_gamma=1.5, accept_fraction=0.1)
    with pytest.raises(ValueError, match="accept_fraction"):
        MixturePolicy(inner=inner, mixture_gamma=0.5, accept_fraction=2.0)


def test_mixture_policy_gamma_one_is_pure_signal():
    """gamma=1 routes every frame to the inner policy."""
    inner = NoFilterPolicy()
    policy = MixturePolicy(
        inner=inner, mixture_gamma=1.0, accept_fraction=0.0, rng_seed=0,
    )
    model, device = _DummyModel(), torch.device("cpu")
    for i in range(50):
        action, meta = policy.select_action(
            _make_item(frame_id=f"f{i}"), model, device,
        )
        assert action == "accept"
        assert meta["path"] == "signal"

    assert policy.count_signal_path == 50
    assert policy.count_random_path == 0
    assert policy.count_accept_signal == 50
    assert policy.count_accept_random == 0


def test_mixture_policy_gamma_zero_is_pure_random():
    """gamma=0 routes every frame to the random path; accept_fraction governs."""
    policy = MixturePolicy(
        inner=NoFilterPolicy(),
        mixture_gamma=0.0,
        accept_fraction=1.0,
        rng_seed=0,
    )
    model, device = _DummyModel(), torch.device("cpu")
    for i in range(20):
        action, meta = policy.select_action(
            _make_item(frame_id=f"f{i}"), model, device,
        )
        assert action == "accept"
        assert meta["path"] == "random"
    assert policy.count_random_path == 20
    assert policy.count_signal_path == 0


def test_mixture_policy_random_accepts_recorded_into_inner_reservoir():
    """Random-path accepts must enter the inner policy's accepted-frame buffer.

    This is the contract that makes the refresh see the actual training
    distribution rather than only the signal-selected subset.
    """
    inner = _make_dist_policy(
        bootstrap_scores=[float(i) for i in range(100)],
        refresh_window_size=20,
    )
    policy = MixturePolicy(
        inner=inner, mixture_gamma=0.0, accept_fraction=1.0, rng_seed=0,
    )
    model, device = _DummyModel(), torch.device("cpu")

    for i in range(15):
        policy.select_action(
            _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}"),
            model, device,
        )
    # All 15 accepts traveled the random path; all should be in the inner window.
    assert inner.get_accepted_frame_ids() == [f"f{i}" for i in range(15)]


def test_mixture_policy_gamma_routing_matches_seed():
    """Same seed -> identical signal/random routing pattern."""
    seq = []
    for _ in range(2):
        policy = MixturePolicy(
            inner=NoFilterPolicy(),
            mixture_gamma=0.5,
            accept_fraction=0.5,
            rng_seed=99,
        )
        actions = []
        for i in range(50):
            _, meta = policy.select_action(
                _make_item(frame_id=f"f{i}"),
                _DummyModel(),
                torch.device("cpu"),
            )
            actions.append(meta["path"])
        seq.append(tuple(actions))
    assert seq[0] == seq[1]


def test_mixture_policy_stats_include_inner():
    inner = RandomPolicy(accept_fraction=1.0)
    policy = MixturePolicy(
        inner=inner, mixture_gamma=0.5, accept_fraction=0.5, rng_seed=0,
    )
    for i in range(10):
        policy.select_action(
            _make_item(frame_id=f"f{i}"), _DummyModel(), torch.device("cpu"),
        )
    stats = policy.get_stats()
    assert stats["count_signal_path"] + stats["count_random_path"] == 10
    assert "inner_accept_fraction" in stats


# ---------------------------------------------------------------------------
# create_filter_policy (config-driven dispatch)
# ---------------------------------------------------------------------------


class _PolicyCfg:
    """Minimal duck-typed config for create_filter_policy."""

    def __init__(self, **kwargs):
        self.filter_policy = kwargs.pop("filter_policy", "none")
        self.accept_fraction = kwargs.pop("accept_fraction", 0.1)
        self.threshold_percentile = kwargs.pop("threshold_percentile", 0.10)
        self.scoring_refresh_window_size = kwargs.pop("scoring_refresh_window_size", 0)
        self.scoring_refresh_reservoir_size = kwargs.pop(
            "scoring_refresh_reservoir_size", 0,
        )
        self.uncertainty_top_k = kwargs.pop("uncertainty_top_k", 5)
        self.uncertainty_score_mode = kwargs.pop("uncertainty_score_mode", "topk_mean")
        self.mixture_gamma = kwargs.pop("mixture_gamma", 0.5)
        self.seed = kwargs.pop("seed", 0)
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_create_filter_policy_none():
    pol = create_filter_policy(_PolicyCfg(filter_policy="none"))
    assert isinstance(pol, NoFilterPolicy)


def test_create_filter_policy_random():
    pol = create_filter_policy(_PolicyCfg(filter_policy="random", accept_fraction=0.3))
    assert isinstance(pol, RandomPolicy)
    assert pol.accept_fraction == pytest.approx(0.3)


def test_create_filter_policy_distribution_requires_bootstrap():
    cfg = _PolicyCfg(filter_policy="distribution")
    with pytest.raises(ValueError, match="bootstrap_mean"):
        create_filter_policy(cfg)


def test_create_filter_policy_distribution_full():
    cfg = _PolicyCfg(
        filter_policy="distribution",
        scoring_refresh_window_size=10,
    )
    pol = create_filter_policy(
        cfg,
        bootstrap_mean=torch.zeros(4),
        bootstrap_cov=torch.eye(4),
        scoring_model=_MagnitudeScorer(embed_dim=4),
        bootstrap_scores=[float(i) for i in range(50)],
    )
    assert isinstance(pol, DistributionBasedPolicy)
    assert pol.refresh_window_size == 10


def test_create_filter_policy_uncertainty_full():
    cfg = _PolicyCfg(filter_policy="uncertainty", uncertainty_top_k=3)
    pol = create_filter_policy(
        cfg,
        scoring_model=_ConstantUncertaintyDetector(),
        bootstrap_scores=[0.1, 0.5, 0.9],
    )
    assert isinstance(pol, DetectionUncertaintyPolicy)


def test_create_filter_policy_mixed_distribution_wraps_inner():
    cfg = _PolicyCfg(
        filter_policy="mixed_distribution",
        mixture_gamma=0.3,
        accept_fraction=0.1,
    )
    pol = create_filter_policy(
        cfg,
        bootstrap_mean=torch.zeros(4),
        bootstrap_cov=torch.eye(4),
        scoring_model=_MagnitudeScorer(embed_dim=4),
        bootstrap_scores=[float(i) for i in range(50)],
    )
    assert isinstance(pol, MixturePolicy)
    assert isinstance(pol.inner, DistributionBasedPolicy)
    assert pol.mixture_gamma == pytest.approx(0.3)


def test_create_filter_policy_unknown_rejected():
    with pytest.raises(ValueError, match="Unknown filter policy"):
        create_filter_policy(_PolicyCfg(filter_policy="banana"))


def test_create_filter_policy_reservoir_seed_override_used():
    """Override flows into the policy's reservoir RNG (federated path)."""
    cfg = _PolicyCfg(
        filter_policy="distribution",
        scoring_refresh_reservoir_size=5,
        seed=0,
    )
    bootstrap_mean = torch.zeros(4)
    bootstrap_cov = torch.eye(4)
    bootstrap_scores = [float(i) for i in range(50)]

    def _build(seed_override: int) -> DistributionBasedPolicy:
        pol = create_filter_policy(
            cfg,
            reservoir_seed_override=seed_override,
            bootstrap_mean=bootstrap_mean,
            bootstrap_cov=bootstrap_cov,
            scoring_model=_MagnitudeScorer(embed_dim=4),
            bootstrap_scores=bootstrap_scores,
        )
        assert isinstance(pol, DistributionBasedPolicy)
        return pol

    a = _build(1)
    b = _build(2)

    # Drive 100 accepts; reservoirs of size 5 differ if seeds differ.
    model, device = _DummyModel(), torch.device("cpu")
    for i in range(100):
        item = _make_item(image=_image_with_mean(99.0), frame_id=f"f{i}")
        a.select_action(item, model, device)
        b.select_action(item, model, device)
    assert a.get_accepted_frame_ids() != b.get_accepted_frame_ids()
