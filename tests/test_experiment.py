"""Tests for stream_active_fl.experiment (config loading, run setup)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from stream_active_fl.experiment import (
    create_run_dir,
    load_dataclass_config,
    resolve_manifest_path,
    setup_run_dir,
)


# ---------------------------------------------------------------------------
# load_dataclass_config
# ---------------------------------------------------------------------------


@dataclass
class _DummyConfig:
    name: str = "default"
    lr: float = 1e-3
    epochs: int = 10


def test_load_dataclass_config_valid(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("name: test\nlr: 0.01\nepochs: 5\n")

    result = load_dataclass_config(_DummyConfig, cfg)
    assert result.name == "test"
    assert result.lr == pytest.approx(0.01)
    assert result.epochs == 5


def test_load_dataclass_config_partial(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("name: partial\n")

    result = load_dataclass_config(_DummyConfig, cfg)
    assert result.name == "partial"
    assert result.lr == pytest.approx(1e-3)  # default
    assert result.epochs == 10  # default


def test_load_dataclass_config_unknown_key(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("name: test\ntypo_key: 99\n")

    with pytest.raises(ValueError, match="Unknown config keys"):
        load_dataclass_config(_DummyConfig, cfg)


def test_load_dataclass_config_empty_file(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")

    result = load_dataclass_config(_DummyConfig, cfg)
    assert result == _DummyConfig()


def test_load_dataclass_config_rejects_non_dataclass():
    with pytest.raises(TypeError, match="not a dataclass"):
        load_dataclass_config(dict, "/dev/null")


# ---------------------------------------------------------------------------
# resolve_manifest_path
# ---------------------------------------------------------------------------


def test_resolve_manifest_path_relative(tmp_path: Path):
    result = resolve_manifest_path(tmp_path, "data/manifest.json")
    assert result == tmp_path / "data/manifest.json"


def test_resolve_manifest_path_absolute(tmp_path: Path):
    abs_path = Path("/absolute/manifest.json")
    result = resolve_manifest_path(tmp_path, abs_path)
    assert result == abs_path


# ---------------------------------------------------------------------------
# create_run_dir / setup_run_dir
# ---------------------------------------------------------------------------


def test_create_run_dir(tmp_path: Path):
    run_dir = create_run_dir(tmp_path)
    assert run_dir.exists()
    assert run_dir.parent == tmp_path


def test_setup_run_dir_copies_config(tmp_path: Path):
    config_file = tmp_path / "my_config.yaml"
    config_file.write_text("key: value\n")

    run_dir = setup_run_dir(tmp_path, "outputs", config_file)
    assert run_dir.exists()
    assert (run_dir / "config.yaml").read_text() == "key: value\n"
