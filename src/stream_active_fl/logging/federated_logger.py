"""
Metrics tracking for federated streaming experiments.

Logs per-round server evaluation metrics and per-client training counts
to a single CSV file (rounds.csv).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


# Column definitions per task
_CLASSIFICATION_EVAL_COLS = [
    "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1",
]
_CLASSIFICATION_EVAL_KEYS = ["loss", "accuracy", "precision", "recall", "f1"]

_DETECTION_EVAL_COLS = [
    "eval_mAP", "eval_mAP_50", "eval_mAP_75",
    "eval_AP_person", "eval_AP_car", "eval_AP_traffic_light",
]
_DETECTION_EVAL_KEYS = [
    "mAP", "mAP_50", "mAP_75", "AP_person", "AP_car", "AP_traffic_light",
]


class FederatedMetricsLogger:
    """CSV logger for per-round federated experiment metrics.

    Writes rounds.csv with one row per federated round, containing
    server-side evaluation metrics and per-client training counts.

    Args:
        log_dir: Directory to write rounds.csv into.
        num_clients: Number of clients (determines per-client columns).
        task: "classification" or "detection".  Controls which evaluation
            metric columns are used.
    """

    def __init__(
        self,
        log_dir: str | Path,
        num_clients: int,
        task: str = "classification",
    ):
        self.log_dir = Path(log_dir)
        self.task = task

        if task == "detection":
            self._eval_cols = _DETECTION_EVAL_COLS
            self._eval_keys = _DETECTION_EVAL_KEYS
        else:
            self._eval_cols = _CLASSIFICATION_EVAL_COLS
            self._eval_keys = _CLASSIFICATION_EVAL_KEYS

        self.rounds_file = self.log_dir / "rounds.csv"
        with open(self.rounds_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["round", "elapsed_seconds"] + list(self._eval_cols)
            for c in range(num_clients):
                header += [f"client_{c}_items", f"client_{c}_trained"]
            writer.writerow(header)

    def log_round(
        self,
        round_idx: int,
        eval_metrics: Optional[dict],
        client_results: list[dict],
        elapsed: float,
    ) -> None:
        """Append one row to rounds.csv.

        Args:
            round_idx: Current federated round number.
            eval_metrics: Server evaluation dict, or None if evaluation
                was skipped this round.
            client_results: Per-client dicts with items_processed and
                items_trained counts.
            elapsed: Wall-clock seconds since training started.
        """
        row: list = [round_idx, f"{elapsed:.1f}"]
        if eval_metrics is not None:
            row += [f"{eval_metrics.get(k, 0.0):.4f}" for k in self._eval_keys]
        else:
            row += [""] * len(self._eval_keys)
        for cr in client_results:
            row += [cr["items_processed"], cr["items_trained"]]

        with open(self.rounds_file, "a", newline="") as f:
            csv.writer(f).writerow(row)
