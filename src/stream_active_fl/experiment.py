"""
Experiment infrastructure: config loading, run setup, model building.

Consolidates helpers shared across all experiment scripts (offline,
streaming, federated) so each script focuses on its own pipeline logic.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml

from stream_active_fl.models import Detector

T = TypeVar("T")


# =============================================================================
# Config loading
# =============================================================================


def load_dataclass_config(config_cls: type[T], path: str | Path) -> T:
    """
    Load a YAML config into a dataclass with strict unknown-key validation.

    Raises:
        TypeError: If config_cls is not a dataclass type.
        ValueError: If YAML root is not a mapping or contains unknown keys.
    """
    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls} is not a dataclass type.")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping of key/value pairs.")

    allowed = set(config_cls.__dataclass_fields__.keys())
    unknown = sorted(set(data.keys()) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {unknown}. "
            f"Please remove typos or add these fields to {config_cls.__name__}."
        )

    return config_cls(**data)


# =============================================================================
# Run setup
# =============================================================================


def create_run_dir(base_output_dir: Path) -> Path:
    """Create a timestamped run directory under base_output_dir."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_run_dir(project_root: Path, output_dir: str | Path, config_path: Path) -> Path:
    """Create run directory under output_dir and copy config there."""
    run_dir = create_run_dir(project_root / Path(output_dir))
    shutil.copy(config_path, run_dir / "config.yaml")
    return run_dir


def resolve_manifest_path(project_root: Path, manifest_path: str | Path) -> Path:
    """
    Resolve manifest path with optional shared-data override.

    Resolution order:
      1) Absolute path as-is.
      2) Path relative to project_root.
      3) For paths under "data/", map to $STREAM_ACTIVE_FL_DATA_ROOT if set.
    """
    resolved = Path(manifest_path)
    if not resolved.is_absolute():
        project_candidate = project_root / resolved
        if project_candidate.exists():
            return project_candidate

        if resolved.parts and resolved.parts[0] == "data":
            data_root = os.environ.get("STREAM_ACTIVE_FL_DATA_ROOT")
            if data_root:
                return Path(data_root) / Path(*resolved.parts[1:])

        return project_candidate
    return resolved


# =============================================================================
# Model building
# =============================================================================


def build_detector_from_config(config: Any) -> Detector:
    """
    Build a Detector from any config object with model-related fields.

    Expected attributes: num_classes, trainable_backbone_layers,
    image_min_size, image_max_size, pretrained_backbone, pretrained_detector.
    """
    return Detector(
        num_classes=config.num_classes,
        trainable_backbone_layers=config.trainable_backbone_layers,
        image_min_size=config.image_min_size,
        image_max_size=config.image_max_size,
        pretrained_backbone=config.pretrained_backbone,
        pretrained_detector=config.pretrained_detector,
    )
