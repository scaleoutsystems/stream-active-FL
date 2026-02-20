"""
Client-sequence partitioning for federated learning simulation.

Partitions a StreamingDataset's sequences into disjoint subsets, one per
simulated client, and provides a resumable stream iterator per client.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Optional

from .datasets import StreamingDataset
from .items import StreamItem


# =============================================================================
# Partitioning
# =============================================================================


def partition_sequences(
    num_sequences: int,
    num_clients: int,
    strategy: str = "uniform",
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition sequence indices into disjoint per-client subsets.

    Args:
        num_sequences: Total number of sequences in the dataset.
        num_clients: Number of federated clients.
        strategy: "uniform" shuffles and splits evenly; "contiguous"
            assigns contiguous blocks (preserves temporal ordering
            within and across clients).
        seed: Random seed for shuffling (only used with "uniform").

    Returns:
        List of num_clients lists, each containing the sequence indices
        assigned to that client.  Subsets are disjoint and their union
        covers all sequences.

    Raises:
        ValueError: If num_clients exceeds num_sequences.
    """
    if num_clients > num_sequences:
        raise ValueError(
            f"num_clients ({num_clients}) exceeds num_sequences ({num_sequences})"
        )

    indices = list(range(num_sequences))

    if strategy == "uniform":
        rng = random.Random(seed)
        rng.shuffle(indices)
    elif strategy == "contiguous":
        pass  # keep original order
    else:
        raise ValueError(f"Unknown partition strategy: {strategy!r}")

    # Split as evenly as possible; first (num_sequences % num_clients)
    # clients get one extra sequence.
    partitions: List[List[int]] = []
    base_size = num_sequences // num_clients
    remainder = num_sequences % num_clients
    offset = 0
    for i in range(num_clients):
        size = base_size + (1 if i < remainder else 0)
        partitions.append(indices[offset : offset + size])
        offset += size

    return partitions


# =============================================================================
# Client stream
# =============================================================================


class ClientStream:
    """
    A resumable stream over a client's assigned sequences.

    Wraps a StreamingDataset and a list of sequence indices to produce a
    single stream of StreamItem objects in temporal order.  The stream is
    resumable: partially consuming it (e.g. with max_items in the training
    loop) and then iterating again continues from where it left off.

    Args:
        dataset: The underlying StreamingDataset (shared, read-only).
        seq_indices: Sequence indices assigned to this client.
    """

    def __init__(self, dataset: StreamingDataset, seq_indices: List[int]):
        self.dataset = dataset
        self.seq_indices = seq_indices
        self._total_items = sum(
            dataset.sequence_metadata[i]["num_subsampled_frames"]
            for i in seq_indices
        )
        self._iterator: Optional[Iterator[StreamItem]] = None
        self._exhausted = False
        self.items_yielded = 0

    @property
    def total_items(self) -> int:
        """Total number of items in this client's stream."""
        return self._total_items

    @property
    def remaining_items(self) -> int:
        """Estimated items remaining (may differ slightly due to failed reads)."""
        return self._total_items - self.items_yielded

    @property
    def exhausted(self) -> bool:
        """Whether the stream has been fully consumed."""
        return self._exhausted

    def _generate(self) -> Iterator[StreamItem]:
        for seq_idx in self.seq_indices:
            for item in self.dataset.get_sequence_iterator(seq_idx):
                self.items_yielded += 1
                yield item
        self._exhausted = True

    def __iter__(self) -> Iterator[StreamItem]:
        """Return a resumable iterator over this client's items.

        Always returns the same internal iterator so that partially
        consuming the stream (e.g. via break) and then iterating again
        continues from the right place.
        """
        if self._iterator is None:
            self._iterator = self._generate()
        return self._iterator

    def __next__(self) -> StreamItem:
        if self._iterator is None:
            self._iterator = self._generate()
        return next(self._iterator)

    def get_sequence_stats(self) -> List[Dict[str, Any]]:
        """Per-sequence classification statistics for this client's partition.

        Useful for diagnosing data heterogeneity across clients (non-IID).
        Positive counts are based on the has_target flag in the dataset's
        per-frame metadata, so this is only meaningful for classification.

        Returns:
            List of dicts, one per sequence, each with keys seq_idx,
            seq_id, num_frames, num_positive, positive_rate.
        """
        stats = []
        for seq_idx in self.seq_indices:
            meta = self.dataset.sequence_metadata[seq_idx]
            frame_info = meta["frame_info"]
            num_positive = sum(
                1 for fi in range(0, meta["num_frames"], self.dataset.subsample_steps)
                if frame_info.get(fi, {}).get("has_target", False)
            )
            stats.append({
                "seq_idx": seq_idx,
                "seq_id": meta["seq_id"],
                "num_frames": meta["num_subsampled_frames"],
                "num_positive": num_positive,
                "positive_rate": num_positive / max(meta["num_subsampled_frames"], 1),
            })
        return stats

    def __repr__(self) -> str:
        return (
            f"ClientStream(sequences={len(self.seq_indices)}, "
            f"items={self._total_items}, "
            f"yielded={self.items_yielded}, "
            f"exhausted={self._exhausted})"
        )
