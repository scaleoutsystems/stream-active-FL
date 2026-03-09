"""
Training loops for the two-phase streaming detection pipeline.

Phase 1 -- Bootstrap:
    bootstrap_train()          Multi-epoch training on the first N frames
    collect_embeddings()       Extract backbone embeddings for bootstrap frames

Phase 2 -- Streaming:
    train_on_stream()          Single-pass buffer-based streaming training
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..core.items import StreamItem
from ..logging import StreamingMetricsLogger
from ..memory import TrainingBuffer
from ..policies import FilterPolicy


# =============================================================================
# Result types
# =============================================================================


@dataclass
class StreamingTrainResult:
    """Summary returned after streaming training."""

    items_processed: int
    items_accepted: int
    items_rejected: int
    buffer_flushes: int
    optimizer_steps: int


# =============================================================================
# Phase 1: Bootstrap training
# =============================================================================


def bootstrap_train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    max_grad_norm: float = 0.0,
    progress_bar: bool = True,
) -> Tuple[float, int]:
    """
    Multi-epoch training on bootstrap frames (standard supervised detection).

    This is the only place where multi-epoch training is allowed. Uses a
    standard DataLoader with shuffle.

    Args:
        model: The detection model.
        train_loader: DataLoader over DetectionDataset (bootstrap subset).
        optimizer: Optimizer for trainable parameters.
        device: Training device.
        epochs: Number of training epochs.
        max_grad_norm: Gradient clipping norm (0 = disabled).
        progress_bar: Show progress bar.

    Returns:
        (final_epoch_loss, total_steps): Average loss from the last epoch
        and total number of optimizer steps taken.
    """
    model.train()
    total_steps = 0
    epoch_loss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0

        loader = tqdm(train_loader, desc=f"Bootstrap epoch {epoch + 1}/{epochs}") if progress_bar else train_loader

        for batch in loader:
            if batch is None:
                continue

            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )

            optimizer.step()
            total_steps += 1
            running_loss += loss.item()
            n_batches += 1

            if progress_bar and hasattr(loader, "set_postfix"):
                loader.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs} — avg loss: {epoch_loss:.4f}")

    return epoch_loss, total_steps


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    progress_bar: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect backbone embeddings for all frames in a DataLoader.

    Runs a forward pass through each batch, extracting embeddings via
    model.get_embedding(). Computes the mean and covariance of the
    embedding distribution.

    Args:
        model: The detection model with a get_embedding() method.
        data_loader: DataLoader over the frames to embed.
        device: Device to run on.
        progress_bar: Show progress bar.

    Returns:
        (mean, cov): Mean vector (D,) and covariance matrix (D, D).
    """
    all_embeddings: List[torch.Tensor] = []

    loader = tqdm(data_loader, desc="Collecting embeddings") if progress_bar else data_loader

    for batch in loader:
        if batch is None:
            continue
        images, _ = batch
        images = [img.to(device) for img in images]
        emb = model.get_embedding(images)  # (B, D)
        all_embeddings.append(emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)

    mean = embeddings.mean(dim=0)
    centered = embeddings - mean.unsqueeze(0)
    cov = (centered.T @ centered) / max(len(embeddings) - 1, 1)

    return mean, cov


# =============================================================================
# Phase 2: Buffer-based streaming training
# =============================================================================


def train_on_stream(
    model: nn.Module,
    stream: Iterable[StreamItem],
    optimizer: torch.optim.Optimizer,
    filter_policy: FilterPolicy,
    training_buffer: TrainingBuffer,
    device: torch.device,
    *,
    max_grad_norm: float = 0.0,
    train_steps_per_buffer: int = 1,
    metrics_logger: Optional[StreamingMetricsLogger] = None,
    eval_fn: Optional[Callable[[nn.Module], Dict[str, Any]]] = None,
    eval_every_n_checkpoints: int = 1,
    novelty_tracker: Optional[Any] = None,
    progress_bar: bool = True,
    total_items: Optional[int] = None,
) -> StreamingTrainResult:
    """
    Single-pass buffer-based streaming training.

    Processes stream items one at a time. For each item the filter policy
    decides accept or reject.  Accepted items go into the TrainingBuffer.
    When the buffer is full, one (or a few) optimizer steps are performed
    on the full buffer, then the buffer is cleared and streaming continues.

    Args:
        model: The detection model.
        stream: Iterable of StreamItem in chronological order.
        optimizer: Optimizer for trainable parameters.
        filter_policy: Policy that decides accept/reject per item.
        training_buffer: Buffer that accumulates accepted items.
        device: Training device.
        max_grad_norm: Gradient clipping norm (0 = disabled).
        train_steps_per_buffer: How many optimizer steps to run when the
            buffer is full. Usually 1.
        metrics_logger: Optional logger for streaming metrics.
        eval_fn: Optional evaluation callback (model) -> metrics_dict.
        eval_every_n_checkpoints: Evaluate every N checkpoints.
        novelty_tracker: Optional NoveltyTracker for novelty metrics.
        progress_bar: Show progress bar.
        total_items: Total expected items (for progress bar).

    Returns:
        StreamingTrainResult with processing statistics.
    """
    if total_items is None and hasattr(stream, "__len__"):
        total_items = len(stream)

    items_processed = 0
    items_accepted = 0
    items_rejected = 0
    optimizer_steps = 0
    checkpoint_idx = 0

    pbar = tqdm(stream, desc="Streaming", total=total_items) if progress_bar else stream

    for stream_item in pbar:
        items_processed += 1

        # Filter decision
        action, meta = filter_policy.select_action(stream_item, model, device)

        if novelty_tracker is not None:
            novelty_tracker.observe(stream_item.categories, action)

        if action == "accept":
            items_accepted += 1
            training_buffer.add(stream_item)
        else:
            items_rejected += 1

        # Log per-item decision
        if metrics_logger is not None:
            forward_pass = True  # filter policies always inspect the item
            metrics_logger.log_stream_item(action, forward_pass=forward_pass)
            metrics_logger.log_decision(
                global_idx=stream_item.metadata.get("global_idx", items_processed - 1),
                frame_id=stream_item.metadata.get("frame_id", ""),
                action=action,
                filter_score=meta.get(
                    "score",
                    meta.get("uncertainty", meta.get("grad_norm", meta.get("random_score", 0.0))),
                ),
                categories=stream_item.categories,
                is_novel=(novelty_tracker.last_was_novel if novelty_tracker else False),
            )

        # Train when buffer is full
        if training_buffer.is_full():
            model.train()
            images, targets = training_buffer.get_batch()
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            for _ in range(train_steps_per_buffer):
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                loss.backward()

                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )

                optimizer.step()
                optimizer_steps += 1

            training_buffer.clear()

        # Checkpoint and evaluation
        if metrics_logger is not None and metrics_logger.should_checkpoint():
            checkpoint_idx += 1

            filter_stats = filter_policy.get_stats()
            buffer_stats = training_buffer.get_stats()
            novelty_stats = novelty_tracker.get_stats() if novelty_tracker else None
            metrics_logger.log_checkpoint(
                checkpoint_idx,
                optimizer_steps,
                filter_stats,
                buffer_stats,
                novelty_stats,
            )

            selection_stats = filter_policy.get_selection_stats()
            metrics_logger.log_filter_stats(checkpoint_idx, selection_stats)
            filter_policy.reset_selection_stats()

            if eval_fn is not None and checkpoint_idx % eval_every_n_checkpoints == 0:
                eval_metrics = eval_fn(model)
                metrics_logger.log_evaluation(checkpoint_idx, eval_metrics)

                if progress_bar and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({
                        "mAP": f"{eval_metrics.get('mAP', 0.0):.3f}",
                        "accept": f"{filter_stats.get('accept_rate', 1.0):.2f}",
                    })

    return StreamingTrainResult(
        items_processed=items_processed,
        items_accepted=items_accepted,
        items_rejected=items_rejected,
        buffer_flushes=training_buffer.total_flushes,
        optimizer_steps=optimizer_steps,
    )
