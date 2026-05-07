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
from typing import Any, Optional, TypeVar

import yaml

from stream_active_fl.models import Detector

T = TypeVar("T")


# =============================================================================
# Config loading
# =============================================================================


def load_dataclass_config(config_cls: type[T], path: str | Path) -> T:
    """
    Load a YAML config into a dataclass with strict unknown-key validation.

    Unknown keys raise rather than being silently dropped, which catches
    typos in long config files at load time.

    Args:
        config_cls: Target dataclass type.  Its field names define the
            allowed top-level YAML keys.
        path: Path to the YAML file.  The root must be a mapping.

    Returns:
        An instance of `config_cls` populated from the YAML mapping.

    Raises:
        TypeError: If `config_cls` is not a dataclass type.
        ValueError: If the YAML root is not a mapping, or it contains
            keys not declared on `config_cls`.
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


def create_run_dir(base_output_dir: Path, seed: Optional[int] = None) -> Path:
    """Create a timestamped run directory under `base_output_dir`.

    When `seed` is provided, the run is nested under `seed_<N>/` so that
    multi-seed experiments can coexist cleanly under the same base dir:
    `outputs/<exp>/seed_<N>/<timestamp>/`.

    Args:
        base_output_dir: Parent directory; created lazily by this call.
        seed: Optional random seed.  When set, an extra `seed_<N>/`
            level is inserted between `base_output_dir` and the
            timestamp.

    Returns:
        The newly created run directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if seed is not None:
        run_dir = base_output_dir / f"seed_{seed}" / timestamp
    else:
        run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_run_dir(
    project_root: Path,
    output_dir: str | Path,
    config_path: Path,
    seed: Optional[int] = None,
) -> Path:
    """Create a run directory and snapshot the config inside it.

    Convenience wrapper around `create_run_dir`: the run dir is created
    under `project_root / output_dir` (resolving relative paths against
    the project root) and the config file is copied to
    `<run_dir>/config.yaml` so each run is self-describing.

    Args:
        project_root: Repository root; used to resolve a relative
            `output_dir`.
        output_dir: Either an absolute path or a path relative to
            `project_root`.
        config_path: Source YAML file to copy into the run directory.
        seed: Optional random seed forwarded to `create_run_dir` to
            nest the run under `seed_<N>/`.

    Returns:
        The created run directory containing a `config.yaml` snapshot.
    """
    run_dir = create_run_dir(project_root / Path(output_dir), seed=seed)
    shutil.copy(config_path, run_dir / "config.yaml")
    return run_dir


def resolve_manifest_path(project_root: Path, manifest_path: str | Path) -> Path:
    """
    Resolve a manifest path with an optional shared-data override.

    Manifests can live either inside the repo (under `data/`) or on a
    shared filesystem pointed to by `$STREAM_ACTIVE_FL_DATA_ROOT`.  This
    helper centralizes the resolution rule so all entry points behave
    identically.

    Resolution order:
        1. Absolute path: returned as-is.
        2. Existing path relative to `project_root`: returned.
        3. Path that begins with `data/`: rebased onto
           `$STREAM_ACTIVE_FL_DATA_ROOT` when that environment variable
           is set.
        4. Fallback: returned as `project_root / manifest_path`
           (caller will get a clear FileNotFoundError downstream).

    Args:
        project_root: Repository root used as the relative-path anchor.
        manifest_path: Manifest file path as given in the YAML config.

    Returns:
        The resolved manifest path.  No existence check is performed
        for the absolute-path branch.
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
    Build a `Detector` from any config object with model-related fields.

    Duck-typed by design: the streaming, federated, and offline configs
    all share the same model fields, so this helper takes any object
    that exposes them.

    Args:
        config: Any object with the following attributes:
            - `num_classes` (int)
            - `trainable_backbone_layers` (int)
            - `image_min_size` (int)
            - `image_max_size` (int)
            - `pretrained_backbone` (bool)
            - `pretrained_detector` (bool)

    Returns:
        A new `Detector` instance.

    Raises:
        AttributeError: If `config` is missing any required attribute.
    """
    return Detector(
        num_classes=config.num_classes,
        trainable_backbone_layers=config.trainable_backbone_layers,
        image_min_size=config.image_min_size,
        image_max_size=config.image_max_size,
        pretrained_backbone=config.pretrained_backbone,
        pretrained_detector=config.pretrained_detector,
    )
