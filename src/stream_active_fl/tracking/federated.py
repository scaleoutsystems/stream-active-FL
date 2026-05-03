"""
Metrics tracking for federated streaming experiments.

Logs per-round server evaluation metrics and per-client training counts
to a single CSV file (rounds.csv).

Detection columns align with offline (epochs.csv) and streaming (checkpoints.csv):
- Aggregate: mAP, mAP_50, mAP_75
- Counts: num_items, total_predictions, total_ground_truth
- Per-class: AP_Vehicle, AP_VulnerableVehicle, ... (from CATEGORY_ID_TO_NAME)

Per-domain detection metrics are emitted into per_domain_checkpoints.csv
in the same long-format schema as the streaming logger, so the same
downstream analysis modules read either pipeline's output without
modification.

Per-frame filter decisions are written to decisions.csv by
FederatedDecisionsLogger, mirroring the streaming decisions log with
additional round and client_id columns.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from ..core import CATEGORY_ID_TO_NAME
from ..evaluation import EXTENDED_DOMAIN_DIMS

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
        self._class_names = list(names)
        self._per_class_ap_cols = [f"AP_{name}" for name in names]

        if task == "detection":
            self._eval_cols = ["mAP", "mAP_50", "mAP_75"] + _DETECTION_COUNT_COLS + self._per_class_ap_cols
        else:
            self._eval_cols = _CLASSIFICATION_EVAL_COLS

        self.rounds_file = self.log_dir / "rounds.csv"
        with open(self.rounds_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["round", "elapsed_seconds", "items_processed_total",
                      "optimizer_steps_total"] + list(self._eval_cols)
            for c in range(num_clients):
                header += [
                    f"client_{c}_items",
                    f"client_{c}_accepted",
                    f"client_{c}_rejected",
                    f"client_{c}_optimizer_steps",
                ]
            writer.writerow(header)

        # Per-domain checkpoints are emitted in the same long-format schema
        # as the streaming logger so analysis modules can load both
        # pipelines uniformly via load_per_domain_checkpoints().
        self.per_domain_checkpoints_file = self.log_dir / "per_domain_checkpoints.csv"
        self._per_domain_header_written = False

        # Cumulative tallies across all rounds; used as the analogue of
        # streaming's items_processed / optimizer_steps so iso-compute
        # comparisons (which key off optimizer_steps) work uniformly.
        self._items_processed_total: int = 0
        self._optimizer_steps_total: int = 0

    def log_round(
        self,
        round_idx: int,
        eval_metrics: Optional[dict],
        client_results: list[dict],
        elapsed: float,
    ) -> None:
        """Append one row to rounds.csv (and per_domain_checkpoints.csv if relevant)."""
        for cr in client_results:
            self._items_processed_total += int(cr.get("items_processed", 0))
            self._optimizer_steps_total += int(cr.get("optimizer_steps", 0))

        row: list = [
            round_idx,
            f"{elapsed:.1f}",
            self._items_processed_total,
            self._optimizer_steps_total,
        ]
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

        if eval_metrics is not None and self.task == "detection":
            self._log_per_domain(round_idx, elapsed, eval_metrics)

    def _log_per_domain(
        self,
        round_idx: int,
        elapsed: float,
        eval_metrics: Dict[str, float],
    ) -> None:
        """Emit long-format per-(dim, bucket) rows when present in metrics.

        Schema is identical to ``StreamingMetricsLogger._log_per_domain``
        so ``stream_active_fl.analysis.runs.load_per_domain_checkpoints``
        reads either without case logic.  ``checkpoint_idx`` is the
        federated round.
        """
        bucket_keys: List[tuple] = []
        dim_prefixes = sorted(
            [(f"mAP_{d}_", d) for d in EXTENDED_DOMAIN_DIMS],
            key=lambda p: len(p[0]),
            reverse=True,
        )
        for key in eval_metrics:
            for prefix, dim in dim_prefixes:
                if key.startswith(prefix):
                    bucket = key[len(prefix):]
                    if bucket:
                        bucket_keys.append((dim, bucket))
                    break

        if not bucket_keys:
            return

        if not self._per_domain_header_written:
            with open(self.per_domain_checkpoints_file, "w", newline="") as f:
                csv.writer(f).writerow([
                    "checkpoint_idx",
                    "items_processed",
                    "optimizer_steps",
                    "elapsed_seconds",
                    "dimension",
                    "bucket",
                    "n_frames",
                    "mAP",
                    "mAP_50",
                    "mAP_75",
                ] + self._per_class_ap_cols)
            self._per_domain_header_written = True

        with open(self.per_domain_checkpoints_file, "a", newline="") as f:
            writer = csv.writer(f)
            for dim, bucket in sorted(set(bucket_keys)):
                tag = f"{dim}_{bucket}"
                n = eval_metrics.get(f"n_{tag}", 0.0)
                row = [
                    round_idx,
                    self._items_processed_total,
                    self._optimizer_steps_total,
                    f"{elapsed:.2f}",
                    dim,
                    bucket,
                    int(n),
                    f"{eval_metrics.get(f'mAP_{tag}', 0.0):.4f}",
                    f"{eval_metrics.get(f'mAP_50_{tag}', 0.0):.4f}",
                    f"{eval_metrics.get(f'mAP_75_{tag}', 0.0):.4f}",
                ]
                for cls in self._class_names:
                    row.append(f"{eval_metrics.get(f'AP_{cls}_{tag}', 0.0):.4f}")
                writer.writerow(row)


class FederatedDecisionsLogger:
    """Per-frame accept/reject log for federated runs.

    Writes decisions.csv in the run directory with the same schema as the
    streaming logger plus round and client_id columns.  One instance is
    shared across all clients and rounds; log_decision is guarded by a
    lock so concurrent clients do not interleave rows.
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_file = self.log_dir / "decisions.csv"
        self._lock = threading.Lock()
        self._start_time = time.time()
        with open(self.decisions_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "round",
                "client_id",
                "global_idx",
                "elapsed_seconds",
                "frame_id",
                "action",
                "filter_metric",
                "filter_score",
                "filter_threshold",
                "categories",
            ])

    def log_decision(
        self,
        round_idx: int,
        client_id: int,
        global_idx: int,
        frame_id: str,
        action: str,
        filter_metric: str,
        filter_score: float,
        filter_threshold: Optional[float],
        categories: Set[str],
    ) -> None:
        elapsed = time.time() - self._start_time
        row = [
            round_idx,
            client_id,
            global_idx,
            f"{elapsed:.2f}",
            frame_id,
            action,
            filter_metric,
            f"{filter_score:.6f}",
            (f"{filter_threshold:.6f}" if filter_threshold is not None else ""),
            ";".join(sorted(categories)) if categories else "",
        ]
        with self._lock:
            with open(self.decisions_file, "a", newline="") as f:
                csv.writer(f).writerow(row)
