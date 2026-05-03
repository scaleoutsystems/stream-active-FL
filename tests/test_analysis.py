"""Smoke tests for the torch-free analysis package.

Verifies that pure-data helpers in `stream_active_fl.analysis.runs` behave
correctly without requiring an `outputs/` tree on disk or a torch
install.  The pipeline-specific table builders (streaming.py, federated.py)
are exercised end-to-end by the notebook-build CLI; we only smoke-import
them here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stream_active_fl.analysis import runs as ah


# ---------------------------------------------------------------------------
# filter_mode (cfg -> family classifier)
# ---------------------------------------------------------------------------


def test_filter_mode_none():
    assert ah.filter_mode({}) == "none"
    assert ah.filter_mode({"filter_policy": "none"}) == "none"


def test_filter_mode_random_and_uncertainty():
    assert ah.filter_mode({"filter_policy": "random"}) == "random"
    assert ah.filter_mode({"filter_policy": "uncertainty"}) == "uncertainty"


def test_filter_mode_distribution_static():
    cfg = {
        "filter_policy": "distribution",
        "scoring_refresh_every_flushes": 0,
        "scoring_refresh_window_size": 200,
    }
    assert ah.filter_mode(cfg) == "static"


def test_filter_mode_distribution_window():
    cfg = {
        "filter_policy": "distribution",
        "scoring_refresh_every_flushes": 5,
        "scoring_refresh_window_size": 200,
        "scoring_refresh_reservoir_size": 0,
    }
    assert ah.filter_mode(cfg) == "window"


def test_filter_mode_distribution_reservoir_takes_precedence():
    """If both sizes are set, reservoir wins (reservoir is the runtime mode)."""
    cfg = {
        "filter_policy": "distribution",
        "scoring_refresh_every_flushes": 5,
        "scoring_refresh_window_size": 100,
        "scoring_refresh_reservoir_size": 200,
    }
    assert ah.filter_mode(cfg) == "reservoir"


def test_filter_mode_distribution_federated_round_field():
    """Federated runs use scoring_refresh_every_rounds, not _every_flushes."""
    cfg = {
        "filter_policy": "distribution",
        "scoring_refresh_every_rounds": 1,
        "scoring_refresh_window_size": 200,
    }
    assert ah.filter_mode(cfg) == "window"


# ---------------------------------------------------------------------------
# manifest_family (cfg -> manifest tag)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manifest_path,expected",
    [
        ("data/manifest_cityday_curated_boot5000.json", "cityday_curated"),
        ("data/manifest_cityday_road_type.json", "cityday_road_type"),
        ("data/manifest_citymix_conditions.json", "citymix_conditions"),
        ("data/manifest_cityday_temporal.json", "unknown"),
        ("", "unknown"),
    ],
)
def test_manifest_family(manifest_path: str, expected: str):
    assert ah.manifest_family({"manifest_path": manifest_path}) == expected


# ---------------------------------------------------------------------------
# get_bootstrap_size (resolution order)
# ---------------------------------------------------------------------------


def test_get_bootstrap_size_manifest_overrides_config():
    manifest = {"ordering": {"bootstrap_frames": 2000}}
    cfg = {"bootstrap_frames": 5000}
    assert ah.get_bootstrap_size(manifest=manifest, cfg=cfg) == 2000


def test_get_bootstrap_size_falls_back_to_config():
    cfg = {"bootstrap_frames": 5000}
    assert ah.get_bootstrap_size(manifest=None, cfg=cfg) == 5000


def test_get_bootstrap_size_falls_back_to_default():
    assert ah.get_bootstrap_size(default=1234) == 1234


# ---------------------------------------------------------------------------
# find_project_root
# ---------------------------------------------------------------------------


def test_find_project_root_from_within_repo():
    """The test process itself runs from inside the repo, so this should work."""
    root = ah.find_project_root(Path(__file__).resolve())
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "stream_active_fl").is_dir()


def test_find_project_root_raises_when_outside(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Could not locate project root"):
        ah.find_project_root(tmp_path)


# ---------------------------------------------------------------------------
# Run discovery on an empty / missing tree
# ---------------------------------------------------------------------------


def test_discover_runs_missing_dir(tmp_path: Path):
    df = ah.discover_runs(tmp_path / "nonexistent")
    assert df.empty


def test_discover_runs_finds_seeded_layout(tmp_path: Path):
    """Verify the canonical 'pipeline/variant/seed_<N>/<timestamp>/' layout."""
    run_dir = (
        tmp_path / "streaming" / "no_filter_curated"
        / "seed_42" / "2026-01-01_00-00-00"
    )
    run_dir.mkdir(parents=True)

    df = ah.discover_runs(tmp_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["pipeline"] == "streaming"
    assert row["variant"] == "no_filter_curated"
    assert row["seed"] == 42
    assert Path(row["run_dir"]) == run_dir


# ---------------------------------------------------------------------------
# Submodule importability (the analysis package must work without torch)
# ---------------------------------------------------------------------------


def test_analysis_submodules_import():
    """Ensure the four core analysis submodules import without torch."""
    from stream_active_fl.analysis import federated, figures, runs, streaming
    from stream_active_fl.analysis.figures import federated as ff
    from stream_active_fl.analysis.figures import streaming as fs

    assert hasattr(runs, "discover_runs")
    assert hasattr(streaming, "FEATURED_VARIANTS")
    assert hasattr(federated, "FEATURED_VARIANTS")
    assert hasattr(fs, "plot_inventory_scatter")
    assert hasattr(fs, "plot_per_block_routing")
    assert hasattr(ff, "plot_inventory_scatter")
    assert hasattr(ff, "plot_per_block_trajectory")
