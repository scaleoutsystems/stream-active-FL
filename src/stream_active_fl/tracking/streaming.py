"""
Metrics tracking for buffer-based streaming detection experiments.

Tracks accept/reject decisions, buffer flushes, and detection performance.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from ..core import CATEGORY_ID_TO_NAME
from ..evaluation import EXTENDED_DOMAIN_DIMS

_DEFAULT_CLASS_NAMES: List[str] = list(CATEGORY_ID_TO_NAME.values())
_CHECKPOINT_COUNT_COLS: List[str] = ["num_items", "total_predictions", "total_ground_truth"]


class StreamingMetricsLogger:
    """
    Logger for buffer-based streaming experiment metrics.

    Tracks:
    - Decisions: accept/reject counts and rates
    - Compute: forward passes, buffer flushes (training events)
    - Performance: periodic evaluation on held-out data (mAP)
    - Per-frame decisions: detailed log for analysis

    CSV files written:
    - streaming_metrics.csv: Main metrics at each checkpoint
    - checkpoints.csv: Aggregate evaluation metrics at checkpoint intervals
    - per_domain_checkpoints.csv: Per-domain mAP in long format, one row
      per (checkpoint, dimension, bucket).  Covers the marginal axes
      (time_of_day, road_condition, road_type) and the joint stream_block
      label when the manifest ordering strategy is recognized.  Only
      written if eval_metrics contains per-domain keys.
    - filter_stats.csv: Per-category filter selection statistics
    - decisions.csv: Per-frame accept/reject log

    Args:
        log_dir: Directory to save CSV logs.
        checkpoint_interval: How often to log checkpoints (in stream items).
        class_names: Class names for per-class AP columns. Defaults to all
            ZOD classes. Pass ClassMapping.names when using a class subset.
    """

    def __init__(
        self,
        log_dir: str | Path,
        checkpoint_interval: int = 1000,
        class_names: Optional[Sequence[str]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_interval = checkpoint_interval
        self._class_names = list(class_names) if class_names is not None else _DEFAULT_CLASS_NAMES
        self._per_class_ap_cols = [f"AP_{name}" for name in self._class_names]

        self.metrics_file = self.log_dir / "streaming_metrics.csv"
        self.checkpoints_file = self.log_dir / "checkpoints.csv"
        self.per_domain_checkpoints_file = self.log_dir / "per_domain_checkpoints.csv"
        self.filter_stats_file = self.log_dir / "filter_stats.csv"
        self.decisions_file = self.log_dir / "decisions.csv"
        self._per_domain_header_written = False

        self.num_items_processed = 0
        self.num_forward_passes = 0
        self.num_accepted = 0
        self.num_rejected = 0
        self.last_optimizer_steps = 0
        self.start_time = time.time()

        self._init_csvs()

    def _init_csvs(self) -> None:
        with open(self.metrics_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "checkpoint_idx",
                "items_processed",
                "forward_passes",
                "accept_count",
                "reject_count",
                "accept_rate",
                "buffer_flushes",
                "buffer_total_items",
                "optimizer_steps",
                "elapsed_seconds",
                "items_per_second",
                "avg_train_loss",
            ])

        with open(self.checkpoints_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "checkpoint_idx",
                "items_processed",
                "optimizer_steps",
                "mAP",
                "mAP_50",
                "mAP_75",
                "elapsed_seconds",
            ] + _CHECKPOINT_COUNT_COLS + self._per_class_ap_cols)

        with open(self.filter_stats_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "checkpoint_idx",
                "items_processed",
                "accept_count",
                "reject_count",
                "accept_rate",
                "accept_by_category",
                "reject_by_category",
            ])

        with open(self.decisions_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "global_idx",
                "checkpoint_idx",
                "elapsed_seconds",
                "frame_id",
                "action",
                "filter_metric",
                "filter_score",
                "filter_threshold",
                "categories",
            ])

    def log_stream_item(
        self,
        action: str,
        forward_pass: bool = True,
    ) -> None:
        """Log processing of a single stream item."""
        self.num_items_processed += 1
        if forward_pass:
            self.num_forward_passes += 1
        if action == "accept":
            self.num_accepted += 1
        else:
            self.num_rejected += 1

    def log_decision(
        self,
        global_idx: int,
        checkpoint_idx: int,
        frame_id: str,
        action: str,
        filter_metric: str,
        filter_score: float,
        filter_threshold: Optional[float],
        categories: Set[str],
    ) -> None:
        """Log a per-frame accept/reject decision."""
        elapsed = time.time() - self.start_time
        with open(self.decisions_file, "a", newline="") as f:
            csv.writer(f).writerow([
                global_idx,
                checkpoint_idx,
                f"{elapsed:.2f}",
                frame_id,
                action,
                filter_metric,
                f"{filter_score:.6f}",
                (f"{filter_threshold:.6f}" if filter_threshold is not None else ""),
                ";".join(sorted(categories)) if categories else "",
            ])

    def log_checkpoint(
        self,
        checkpoint_idx: int,
        optimizer_steps: int,
        filter_stats: Optional[Dict[str, Any]] = None,
        buffer_stats: Optional[Dict[str, Any]] = None,
        avg_train_loss: Optional[float] = None,
    ) -> None:
        """Log a checkpoint (periodic snapshot of metrics)."""
        elapsed = time.time() - self.start_time
        items_per_sec = self.num_items_processed / max(elapsed, 1e-6)
        total = self.num_accepted + self.num_rejected
        accept_rate = self.num_accepted / max(total, 1)

        buffer_flushes = buffer_stats.get("total_flushes", 0) if buffer_stats else 0
        buffer_total = buffer_stats.get("total_items_added", 0) if buffer_stats else 0

        self.last_optimizer_steps = optimizer_steps

        with open(self.metrics_file, "a", newline="") as f:
            csv.writer(f).writerow([
                checkpoint_idx,
                self.num_items_processed,
                self.num_forward_passes,
                self.num_accepted,
                self.num_rejected,
                f"{accept_rate:.4f}",
                buffer_flushes,
                buffer_total,
                optimizer_steps,
                f"{elapsed:.2f}",
                f"{items_per_sec:.2f}",
                f"{avg_train_loss:.6f}" if avg_train_loss is not None else "",
            ])

    def log_filter_stats(
        self,
        checkpoint_idx: int,
        selection_stats: Dict[str, Any],
    ) -> None:
        """Log per-category filter selection stats."""
        accept_by_cat = selection_stats.get("accept_by_category", {})
        reject_by_cat = selection_stats.get("reject_by_category", {})
        with open(self.filter_stats_file, "a", newline="") as f:
            csv.writer(f).writerow([
                checkpoint_idx,
                self.num_items_processed,
                selection_stats.get("accept_count", 0),
                selection_stats.get("reject_count", 0),
                f"{selection_stats.get('accept_rate', 0.0):.4f}",
                json.dumps(accept_by_cat, sort_keys=True),
                json.dumps(reject_by_cat, sort_keys=True),
            ])

    def log_evaluation(
        self,
        checkpoint_idx: int,
        eval_metrics: Dict[str, float],
    ) -> None:
        """Log evaluation metrics at a checkpoint.

        Writes one row to checkpoints.csv (aggregate) and, if the metrics
        dict contains per-domain keys of the form mAP_<dim>_<bucket>, one
        long-format row per (dim, bucket) to per_domain_checkpoints.csv.
        """
        elapsed = time.time() - self.start_time

        row = [
            checkpoint_idx,
            self.num_items_processed,
            self.last_optimizer_steps,
            f"{eval_metrics.get('mAP', 0.0):.4f}",
            f"{eval_metrics.get('mAP_50', 0.0):.4f}",
            f"{eval_metrics.get('mAP_75', 0.0):.4f}",
            f"{elapsed:.2f}",
        ]
        for key in _CHECKPOINT_COUNT_COLS:
            row.append(f"{eval_metrics.get(key, 0.0):.4f}")
        for key in self._per_class_ap_cols:
            row.append(f"{eval_metrics.get(key, 0.0):.4f}")

        with open(self.checkpoints_file, "a", newline="") as f:
            csv.writer(f).writerow(row)

        self._log_per_domain(checkpoint_idx, elapsed, eval_metrics)

    def _log_per_domain(
        self,
        checkpoint_idx: int,
        elapsed: float,
        eval_metrics: Dict[str, float],
    ) -> None:
        """Emit long-format per-(dim, bucket) rows when present in metrics."""
        # Match keys mAP_<dim>_<bucket> against known domain dimensions.
        # Dim names can contain underscores (time_of_day, stream_block);
        # we match longest prefix first so e.g. mAP_road_type_city is
        # parsed as dim=road_type + bucket=city rather than dim=road +
        # bucket=type_city.
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
                    checkpoint_idx,
                    self.num_items_processed,
                    self.last_optimizer_steps,
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

    def should_checkpoint(self) -> bool:
        return self.num_items_processed % self.checkpoint_interval == 0

    def get_summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        total = self.num_accepted + self.num_rejected
        return {
            "items_processed": self.num_items_processed,
            "forward_passes": self.num_forward_passes,
            "accepted": self.num_accepted,
            "rejected": self.num_rejected,
            "accept_rate": self.num_accepted / max(total, 1),
            "optimizer_steps": self.last_optimizer_steps,
            "elapsed_seconds": elapsed,
            "items_per_second": self.num_items_processed / max(elapsed, 1e-6),
        }

    def print_summary(self) -> None:
        stats = self.get_summary()
        print()
        print("=" * 60)
        print("Streaming Metrics Summary")
        print("=" * 60)
        print(f"  Items processed  : {stats['items_processed']}")
        print(f"  Forward passes   : {stats['forward_passes']}")
        print(f"  Accepted         : {stats['accepted']}")
        print(f"  Rejected         : {stats['rejected']}")
        print(f"  Accept rate      : {stats['accept_rate']:.4f}")
        print(f"  Optimizer steps  : {stats['optimizer_steps']}")
        print(f"  Elapsed time     : {stats['elapsed_seconds']:.1f}s")
        print(f"  Items per second : {stats['items_per_second']:.2f}")
        print("=" * 60)
        print()
