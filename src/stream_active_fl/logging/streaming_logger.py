"""
Metrics tracking for streaming experiments.

Tracks compute costs (forward/backward passes), performance over time,
and resource usage.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional


class StreamingMetricsLogger:
    """
    Logger for streaming experiment metrics.

    Tracks:
    - Compute: forward passes, backward passes (updates), wall-clock time
    - Performance: periodic evaluation on held-out data
    - Buffer stats: replay buffer utilization
    - Filter stats: train/store/skip counts and rates

    Args:
        log_dir: Directory to save CSV logs.
        checkpoint_interval: How often to log checkpoints (in stream items processed).
    """

    def __init__(
        self,
        log_dir: str | Path,
        checkpoint_interval: int = 1000,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_interval = checkpoint_interval

        # CSV files
        self.metrics_file = self.log_dir / "streaming_metrics.csv"
        self.checkpoints_file = self.log_dir / "checkpoints.csv"
        self.filter_stats_file = self.log_dir / "filter_stats.csv"

        # Counters
        self.num_items_processed = 0
        self.num_forward_passes = 0
        self.num_backward_passes = 0
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time

        # Initialize CSV files
        self._init_metrics_csv()
        self._init_checkpoints_csv()
        self._init_filter_stats_csv()

    def _init_metrics_csv(self) -> None:
        """Initialize main metrics CSV with headers."""
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "checkpoint_idx",
                "items_processed",
                "forward_passes",
                "backward_passes",
                "train_rate",
                "elapsed_seconds",
                "items_per_second",
                "buffer_size",
                "buffer_utilization",
                "buffer_n_positive",
                "buffer_n_negative",
                "buffer_positive_ratio",
            ])

    def _init_filter_stats_csv(self) -> None:
        """Initialize filter selection stats CSV with headers."""
        with open(self.filter_stats_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "checkpoint_idx",
                "items_processed",
                "train_pos",
                "train_neg",
                "skip_pos",
                "skip_neg",
                "store_pos",
                "store_neg",
                "train_total",
                "train_pos_ratio",
                "pos_train_rate",
                "neg_train_rate",
                "avg_loss_train_pos",
                "avg_loss_train_neg",
                "avg_loss_skip_pos",
                "avg_loss_skip_neg",
            ])

    def _init_checkpoints_csv(self) -> None:
        """Initialize checkpoints CSV with headers (for evaluation metrics)."""
        with open(self.checkpoints_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "checkpoint_idx",
                "items_processed",
                "eval_loss",
                "eval_accuracy",
                "eval_precision",
                "eval_recall",
                "eval_f1",
                "elapsed_seconds",
            ])

    def log_stream_item(
        self,
        action: str,
        forward_pass: bool = True,
        backward_pass: bool = False,
    ) -> None:
        """
        Log processing of a single stream item.

        Args:
            action: The action taken ("train", "store", or "skip").
            forward_pass: Whether a forward pass was performed.
            backward_pass: Whether a backward pass was performed.
        """
        self.num_items_processed += 1
        
        if forward_pass:
            self.num_forward_passes += 1
        
        if backward_pass:
            self.num_backward_passes += 1

    def log_checkpoint(
        self,
        checkpoint_idx: int,
        buffer_stats: Optional[Dict[str, Any]] = None,
        filter_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a checkpoint (periodic snapshot of metrics).

        Args:
            checkpoint_idx: Checkpoint index.
            buffer_stats: Replay buffer statistics.
            filter_stats: Filter policy statistics.
        """
        elapsed = time.time() - self.start_time
        items_per_sec = self.num_items_processed / max(elapsed, 1e-6)
        
        train_rate = (
            self.num_backward_passes / max(self.num_items_processed, 1)
        )

        # Extract buffer stats
        buffer_size = buffer_stats.get("size", 0) if buffer_stats else 0
        buffer_util = buffer_stats.get("utilization", 0.0) if buffer_stats else 0.0
        buffer_n_pos = buffer_stats.get("n_positive", 0) if buffer_stats else 0
        buffer_n_neg = buffer_stats.get("n_negative", 0) if buffer_stats else 0
        buffer_pos_ratio = buffer_stats.get("positive_ratio", 0.0) if buffer_stats else 0.0

        # Write to CSV
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                checkpoint_idx,
                self.num_items_processed,
                self.num_forward_passes,
                self.num_backward_passes,
                f"{train_rate:.4f}",
                f"{elapsed:.2f}",
                f"{items_per_sec:.2f}",
                buffer_size,
                f"{buffer_util:.4f}",
                buffer_n_pos,
                buffer_n_neg,
                f"{buffer_pos_ratio:.4f}",
            ])

        self.last_checkpoint_time = time.time()

    def log_filter_stats(
        self,
        checkpoint_idx: int,
        selection_stats: Dict[str, Any],
    ) -> None:
        """
        Log per-class filter selection stats for the current interval.

        Args:
            checkpoint_idx: Checkpoint index.
            selection_stats: Dict from FilterPolicy.get_selection_stats().
        """
        with open(self.filter_stats_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                checkpoint_idx,
                self.num_items_processed,
                selection_stats.get("train_pos", 0),
                selection_stats.get("train_neg", 0),
                selection_stats.get("skip_pos", 0),
                selection_stats.get("skip_neg", 0),
                selection_stats.get("store_pos", 0),
                selection_stats.get("store_neg", 0),
                selection_stats.get("train_total", 0),
                f"{selection_stats.get('train_pos_ratio', 0.0):.4f}",
                f"{selection_stats.get('pos_train_rate', 0.0):.4f}",
                f"{selection_stats.get('neg_train_rate', 0.0):.4f}",
                f"{selection_stats.get('avg_loss_train_pos', 0.0):.4f}",
                f"{selection_stats.get('avg_loss_train_neg', 0.0):.4f}",
                f"{selection_stats.get('avg_loss_skip_pos', 0.0):.4f}",
                f"{selection_stats.get('avg_loss_skip_neg', 0.0):.4f}",
            ])

    def log_evaluation(
        self,
        checkpoint_idx: int,
        eval_metrics: Dict[str, float],
    ) -> None:
        """
        Log evaluation metrics at a checkpoint.

        Args:
            checkpoint_idx: Checkpoint index.
            eval_metrics: Dict with loss, accuracy, precision, recall, f1.
        """
        elapsed = time.time() - self.start_time

        with open(self.checkpoints_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                checkpoint_idx,
                self.num_items_processed,
                f"{eval_metrics.get('loss', 0.0):.4f}",
                f"{eval_metrics.get('accuracy', 0.0):.4f}",
                f"{eval_metrics.get('precision', 0.0):.4f}",
                f"{eval_metrics.get('recall', 0.0):.4f}",
                f"{eval_metrics.get('f1', 0.0):.4f}",
                f"{elapsed:.2f}",
            ])

    def should_checkpoint(self) -> bool:
        """Check if it's time to log a checkpoint."""
        return self.num_items_processed % self.checkpoint_interval == 0

    def get_summary(self) -> Dict[str, Any]:
        """Get current summary statistics."""
        elapsed = time.time() - self.start_time
        return {
            "items_processed": self.num_items_processed,
            "forward_passes": self.num_forward_passes,
            "backward_passes": self.num_backward_passes,
            "train_rate": self.num_backward_passes / max(self.num_items_processed, 1),
            "elapsed_seconds": elapsed,
            "items_per_second": self.num_items_processed / max(elapsed, 1e-6),
            "forward_savings": 1.0 - (self.num_forward_passes / max(self.num_items_processed, 1)),
            "backward_savings": 1.0 - (self.num_backward_passes / max(self.num_items_processed, 1)),
        }

    def print_summary(self) -> None:
        """Print a summary of current metrics."""
        stats = self.get_summary()
        
        print()
        print("=" * 60)
        print("Streaming Metrics Summary")
        print("=" * 60)
        print(f"  Items processed      : {stats['items_processed']}")
        print(f"  Forward passes       : {stats['forward_passes']}")
        print(f"  Backward passes      : {stats['backward_passes']}")
        print(f"  Train rate           : {stats['train_rate']:.4f}")
        print(f"  Elapsed time         : {stats['elapsed_seconds']:.1f}s")
        print(f"  Items per second     : {stats['items_per_second']:.2f}")
        print(f"  Backward savings     : {stats['backward_savings']:.2%}")
        print("=" * 60)
        print()
