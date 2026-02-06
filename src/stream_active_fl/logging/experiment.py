"""Experiment tracking and reproducibility utilities."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def get_git_info(repo_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get git commit hash and dirty status.

    Args:
        repo_path: Path to the git repository. If None, uses current directory.

    Returns:
        Dict with 'commit' (str or None) and 'dirty' (bool or None).
    """
    try:
        kwargs = {"capture_output": True, "text": True}
        if repo_path is not None:
            kwargs["cwd"] = repo_path

        commit = subprocess.run(["git", "rev-parse", "HEAD"], **kwargs)
        commit_hash = commit.stdout.strip() if commit.returncode == 0 else None

        status = subprocess.run(["git", "status", "--porcelain"], **kwargs)
        is_dirty = bool(status.stdout.strip()) if status.returncode == 0 else None

        return {"commit": commit_hash, "dirty": is_dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def get_environment_info() -> Dict[str, Any]:
    """
    Gather environment information for reproducibility.

    Returns:
        Dict with hostname, platform, Python version, PyTorch version,
        and CUDA/GPU info if available.
    """
    env_info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["gpu_count"] = torch.cuda.device_count()
        try:
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            env_info["gpu_name"] = None

    return env_info


def create_run_dir(base_output_dir: Path) -> Path:
    """
    Create a timestamped run directory.

    Args:
        base_output_dir: Parent directory for experiment outputs.

    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_info(
    run_dir: Path,
    config: Any,
    command: str,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    best_epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    best_metric_name: str = "val_f1",
    dataset_info: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    repo_path: Optional[Path] = None,
) -> None:
    """
    Save run metadata to JSON file.

    Args:
        run_dir: Directory to save run_info.json.
        config: Configuration object (must have __dict__ attribute).
        command: Command used to run the experiment.
        start_time: Experiment start time.
        end_time: Experiment end time (None if still running).
        best_epoch: Epoch with best metric.
        best_metric: Best metric value achieved.
        best_metric_name: Name of the metric being tracked.
        dataset_info: Optional dataset statistics.
        extra_info: Optional additional info to include.
        repo_path: Path to git repository for commit info.
    """
    git_info = get_git_info(repo_path)
    env_info = get_environment_info()

    run_info = {
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat() if end_time else None,
        "duration_seconds": (end_time - start_time).total_seconds() if end_time else None,
        "command": command,
        "config": config.__dict__ if hasattr(config, "__dict__") else config,
        "environment": env_info,
        "dataset_info": dataset_info,
        "best_epoch": best_epoch,
        f"best_{best_metric_name}": best_metric,
    }

    if extra_info:
        run_info.update(extra_info)

    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
