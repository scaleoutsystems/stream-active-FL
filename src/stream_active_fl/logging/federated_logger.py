"""
Metrics tracking for federated streaming experiments.

Logs per-round server evaluation metrics and per-client training counts
to a single CSV file (rounds.csv).

Detection columns align with offline (epochs.csv) and streaming (checkpoints.csv):
- Aggregate: mAP, mAP_50, mAP_75
- Counts: num_items, total_predictions, total_ground_truth
- Per-class: AP_Vehicle, AP_VulnerableVehicle, ... (from CATEGORY_ID_TO_NAME)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Sequence

from ..core import CATEGORY_ID_TO_NAME

_CLASSIFICATION_EVAL_COLS: List[str] = ["loss", "accuracy", "precision", "recall", "f1"]
_DETECTION_COUNT_COLS: List[str] = ["num_items", "total_predictions", "total_ground_truth"]
_DEFAULT_CLASS_NAMES: List[str] = list(CATEGORY_ID_TO_NAME.values())


class FederatedMetricsLogger:
    """CSV logger for per-round federated experiment metrics.

    Writes rounds.csv with one row per federated round, containing
    server-side evaluation metrics and per-client training counts.

    Args:
        log_dir: Directory to write rounds.csv into.
        num_clients: Number of clients (determines per-client columns).
        task: "classification" or "detection".
        class_names: Class names for per-class AP columns (detection only).
            Defaults to all ZOD classes.
    """

    def __init__(
        self,
        log_dir: str | Path,
        num_clients: int,
        task: str = "classification",
        class_names: Optional[Sequence[str]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.task = task

        names = list(class_names) if class_names is not None else _DEFAULT_CLASS_NAMES
        per_class_cols = [f"AP_{name}" for name in names]

        if task == "detection":
            self._eval_cols = ["mAP", "mAP_50", "mAP_75"] + _DETECTION_COUNT_COLS + per_class_cols
        else:
            self._eval_cols = _CLASSIFICATION_EVAL_COLS

        self.rounds_file = self.log_dir / "rounds.csv"
        with open(self.rounds_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["round", "elapsed_seconds"] + list(self._eval_cols)
            for c in range(num_clients):
                header += [
                    f"client_{c}_items",
                    f"client_{c}_accepted",
                    f"client_{c}_rejected",
                    f"client_{c}_optimizer_steps",
                ]
            writer.writerow(header)

    def log_round(
        self,
        round_idx: int,
        eval_metrics: Optional[dict],
        client_results: list[dict],
        elapsed: float,
    ) -> None:
        """Append one row to rounds.csv."""
        row: list = [round_idx, f"{elapsed:.1f}"]
        if eval_metrics is not None:
            row += [f"{eval_metrics.get(k, 0.0):.4f}" for k in self._eval_cols]
        else:
            row += [""] * len(self._eval_cols)
        for cr in client_results:
            row += [
                cr.get("items_processed", 0),
                cr.get("items_accepted", cr.get("items_trained", 0)),
                cr.get("items_rejected", 0),
                cr.get("optimizer_steps", 0),
            ]

        with open(self.rounds_file, "a", newline="") as f:
            csv.writer(f).writerow(row)
