"""Logging utilities for experiment tracking and reproducibility."""

from __future__ import annotations

import csv
import json
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class MetricsLogger:
    """
    Logger for saving training metrics to CSV.

    Writes metrics incrementally to a CSV file, allowing easy analysis
    and plotting without loading PyTorch checkpoints.
    """

    def __init__(self, run_dir: Path, fieldnames: Optional[List[str]] = None):
        """
        Initialize the metrics logger.

        Args:
            run_dir: Directory to save metrics.csv.
            fieldnames: Column names for the CSV. If None, uses default
                fields for epoch-based training with train/val metrics.
        """
        self.csv_path = run_dir / "metrics.csv"

        if fieldnames is None:
            fieldnames = [
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_precision",
                "train_recall",
                "train_f1",
                "val_loss",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
                "epoch_time_seconds",
            ]

        self.fieldnames = fieldnames

        # Initialize CSV with headers
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """
        Log a row of metrics.

        Args:
            row: Dictionary with metric values. Keys should match fieldnames.
                Missing keys will be written as empty strings.
        """
        # Fill missing fields with empty strings
        full_row = {k: row.get(k, "") for k in self.fieldnames}

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(full_row)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        epoch_time: float,
    ) -> None:
        """
        Log metrics for one epoch (convenience method).

        Args:
            epoch: Current epoch number.
            train_metrics: Training metrics dict with loss, accuracy, precision, recall, f1.
            val_metrics: Validation metrics dict (or None if not evaluated).
            epoch_time: Time taken for the epoch in seconds.
        """
        row = {
            "epoch": epoch,
            "train_loss": train_metrics.get("loss", ""),
            "train_accuracy": train_metrics.get("accuracy", ""),
            "train_precision": train_metrics.get("precision", ""),
            "train_recall": train_metrics.get("recall", ""),
            "train_f1": train_metrics.get("f1", ""),
            "val_loss": val_metrics["loss"] if val_metrics else "",
            "val_accuracy": val_metrics["accuracy"] if val_metrics else "",
            "val_precision": val_metrics["precision"] if val_metrics else "",
            "val_recall": val_metrics["recall"] if val_metrics else "",
            "val_f1": val_metrics["f1"] if val_metrics else "",
            "epoch_time_seconds": round(epoch_time, 2),
        }
        self.log(row)
