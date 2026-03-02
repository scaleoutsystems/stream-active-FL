"""
Metrics tracking for buffer-based streaming detection experiments.

Tracks accept/reject decisions, buffer flushes, detection performance,
and novelty metrics.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set


class StreamingMetricsLogger:
    """
    Logger for buffer-based streaming experiment metrics.

    Tracks:
    - Decisions: accept/reject counts and rates
    - Compute: forward passes, buffer flushes (training events)
    - Performance: periodic evaluation on held-out data (mAP)
    - Novelty: novel-category accept rates (from NoveltyTracker)
    - Per-frame decisions: detailed log for analysis

    CSV files written:
    - streaming_metrics.csv: Main metrics at each checkpoint
    - checkpoints.csv: Evaluation metrics at checkpoint intervals
    - filter_stats.csv: Per-category filter selection statistics
    - decisions.csv: Per-frame accept/reject log

    Args:
        log_dir: Directory to save CSV logs.
        checkpoint_interval: How often to log checkpoints (in stream items).
    """

    def __init__(
        self,
        log_dir: str | Path,
        checkpoint_interval: int = 1000,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_interval = checkpoint_interval

        self.metrics_file = self.log_dir / "streaming_metrics.csv"
        self.checkpoints_file = self.log_dir / "checkpoints.csv"
        self.filter_stats_file = self.log_dir / "filter_stats.csv"
        self.decisions_file = self.log_dir / "decisions.csv"

        self.num_items_processed = 0
        self.num_forward_passes = 0
        self.num_accepted = 0
        self.num_rejected = 0
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
                "elapsed_seconds",
                "items_per_second",
                "novel_accept_rate",
                "redundant_reject_rate",
            ])

        with open(self.checkpoints_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "checkpoint_idx",
                "items_processed",
                "eval_mAP",
                "eval_mAP_50",
                "eval_mAP_75",
                "elapsed_seconds",
            ])

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
                "frame_id",
                "action",
                "filter_score",
                "categories",
                "is_novel",
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
        frame_id: str,
        action: str,
        filter_score: float,
        categories: Set[str],
        is_novel: bool,
    ) -> None:
        """Log a per-frame accept/reject decision."""
        with open(self.decisions_file, "a", newline="") as f:
            csv.writer(f).writerow([
                global_idx,
                frame_id,
                action,
                f"{filter_score:.6f}",
                ";".join(sorted(categories)) if categories else "",
                int(is_novel),
            ])

    def log_checkpoint(
        self,
        checkpoint_idx: int,
        filter_stats: Optional[Dict[str, Any]] = None,
        buffer_stats: Optional[Dict[str, Any]] = None,
        novelty_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a checkpoint (periodic snapshot of metrics)."""
        elapsed = time.time() - self.start_time
        items_per_sec = self.num_items_processed / max(elapsed, 1e-6)
        total = self.num_accepted + self.num_rejected
        accept_rate = self.num_accepted / max(total, 1)

        buffer_flushes = buffer_stats.get("total_flushes", 0) if buffer_stats else 0
        buffer_total = buffer_stats.get("total_items_added", 0) if buffer_stats else 0

        novel_accept_rate = novelty_stats.get("novel_accept_rate", 0.0) if novelty_stats else 0.0
        redundant_reject_rate = novelty_stats.get("redundant_reject_rate", 0.0) if novelty_stats else 0.0

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
                f"{elapsed:.2f}",
                f"{items_per_sec:.2f}",
                f"{novel_accept_rate:.4f}",
                f"{redundant_reject_rate:.4f}",
            ])

    def log_filter_stats(
        self,
        checkpoint_idx: int,
        selection_stats: Dict[str, Any],
    ) -> None:
        """Log per-category filter selection stats."""
        with open(self.filter_stats_file, "a", newline="") as f:
            csv.writer(f).writerow([
                checkpoint_idx,
                self.num_items_processed,
                selection_stats.get("accept_count", 0),
                selection_stats.get("reject_count", 0),
                f"{selection_stats.get('accept_rate', 0.0):.4f}",
                str(selection_stats.get("accept_by_category", {})),
                str(selection_stats.get("reject_by_category", {})),
            ])

    def log_evaluation(
        self,
        checkpoint_idx: int,
        eval_metrics: Dict[str, float],
    ) -> None:
        """Log evaluation metrics at a checkpoint."""
        elapsed = time.time() - self.start_time

        with open(self.checkpoints_file, "a", newline="") as f:
            csv.writer(f).writerow([
                checkpoint_idx,
                self.num_items_processed,
                f"{eval_metrics.get('mAP', 0.0):.4f}",
                f"{eval_metrics.get('mAP_50', 0.0):.4f}",
                f"{eval_metrics.get('mAP_75', 0.0):.4f}",
                f"{elapsed:.2f}",
            ])

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
        print(f"  Elapsed time     : {stats['elapsed_seconds']:.1f}s")
        print(f"  Items per second : {stats['items_per_second']:.2f}")
        print("=" * 60)
        print()
