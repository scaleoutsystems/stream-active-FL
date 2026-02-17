"""Metrics logger for epoch-based offline training."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


class OfflineMetricsLogger:
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
