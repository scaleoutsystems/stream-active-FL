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
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sized, Tuple

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
    desc_prefix: str = "Bootstrap",
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[List[Dict[str, float]], int]:
    """
    Multi-epoch training on bootstrap frames (standard supervised detection).

    Also used by the offline baseline for epoch-level training with the same
    loop logic (called with epochs=1 per outer epoch).

    Args:
        desc_prefix: Label shown in the tqdm progress bar (e.g. "Epoch" for
            offline training, "Bootstrap" for streaming bootstrap phase).
        scaler: Optional GradScaler for AMP mixed-precision training.
            When provided, forward passes run under torch.cuda.amp.autocast.

    Returns:
        (epoch_logs, total_steps): Per-epoch metrics and total optimizer steps.
        Each entry in epoch_logs is {"epoch": int, "avg_loss": float, "batches": int}.
    """
    model.train()
    total_steps = 0
    epoch_logs: List[Dict[str, float]] = []
    use_amp = scaler is not None and scaler.is_enabled()

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"{desc_prefix} epoch {epoch + 1}/{epochs}") if progress_bar else None
        loader: Iterable = pbar if pbar is not None else train_loader

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
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                loss = torch.stack(list(loss_dict.values())).sum()

            if scaler is not None:
                scaled_loss = scaler.scale(loss)
                assert isinstance(scaled_loss, torch.Tensor)
                scaled_loss.backward()
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
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

            if pbar is not None:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(n_batches, 1)
        epoch_logs.append({"epoch": epoch + 1, "avg_loss": avg_loss, "batches": n_batches})
        print(f"  Epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}")

    return epoch_logs, total_steps


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    progress_bar: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
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
        (mean, cov, count): Mean vector (D,), covariance matrix (D, D),
        and number of embeddings used.
    """
    was_training = model.training
    model.eval()

    all_embeddings: List[torch.Tensor] = []

    loader = tqdm(data_loader, desc="Collecting embeddings") if progress_bar else data_loader

    for batch in loader:
        if batch is None:
            continue
        images, _ = batch
        images = [img.to(device) for img in images]
        emb = model.get_embedding(images)  # (B, D)
        all_embeddings.append(emb.cpu())

    if was_training:
        model.train()

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)

    mean = embeddings.mean(dim=0)
    centered = embeddings - mean.unsqueeze(0)
    cov = (centered.T @ centered) / max(len(embeddings) - 1, 1)

    return mean, cov, int(len(embeddings))


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
    buffer_training_mode: Literal["full_batch", "mini_batch"] = "full_batch",
    local_epochs_per_buffer: int = 1,
    mini_batch_size: int = 8,
    shuffle_buffer_each_epoch: bool = True,
    metrics_logger: Optional[StreamingMetricsLogger] = None,
    eval_fn: Optional[Callable[[nn.Module], Dict[str, Any]]] = None,
    eval_every_n_checkpoints: int = 1,
    novelty_tracker: Optional[Any] = None,
    progress_bar: bool = True,
    total_items: Optional[int] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> StreamingTrainResult:
    """
    Single-pass buffer-based streaming training.

    Processes stream items one at a time. For each item the filter policy
    decides accept or reject. Accepted items go into the TrainingBuffer.
    When the buffer is full, local training is performed and then the
    buffer is cleared before streaming continues.

    Args:
        model: The detection model.
        stream: Iterable of StreamItem in chronological order.
        optimizer: Optimizer for trainable parameters.
        filter_policy: Policy that decides accept/reject per item.
        training_buffer: Buffer that accumulates accepted items.
        device: Training device.
        max_grad_norm: Gradient clipping norm (0 = disabled).
        train_steps_per_buffer: Number of repeated full-batch optimizer
            steps when buffer_training_mode="full_batch".
        buffer_training_mode:
            - "full_batch": current behavior, repeat steps on entire buffer.
            - "mini_batch": run local epochs over mini-batches from buffer.
        local_epochs_per_buffer: Number of local epochs when
            buffer_training_mode="mini_batch".
        mini_batch_size: Mini-batch size when buffer_training_mode="mini_batch".
        shuffle_buffer_each_epoch: Reshuffle buffer items each local epoch
            in mini-batch mode.
        metrics_logger: Optional logger for streaming metrics.
        eval_fn: Optional evaluation callback (model) -> metrics_dict.
        eval_every_n_checkpoints: Evaluate every N checkpoints.
        novelty_tracker: Optional NoveltyTracker for novelty metrics.
        progress_bar: Show progress bar.
        total_items: Total expected items (for progress bar).

    Returns:
        StreamingTrainResult with processing statistics.
    """
    if total_items is None and isinstance(stream, Sized):
        total_items = len(stream)
    if train_steps_per_buffer < 1:
        raise ValueError("train_steps_per_buffer must be >= 1")
    if local_epochs_per_buffer < 1:
        raise ValueError("local_epochs_per_buffer must be >= 1")
    if mini_batch_size < 1:
        raise ValueError("mini_batch_size must be >= 1")

    items_processed = 0
    items_accepted = 0
    items_rejected = 0
    optimizer_steps = 0
    checkpoint_idx = 0

    use_amp = scaler is not None and scaler.is_enabled()

    running_train_loss = 0.0
    train_loss_steps = 0

    def _optimizer_step(
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        nonlocal optimizer_steps, running_train_loss, train_loss_steps
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model.train()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            loss = torch.stack(list(loss_dict.values())).sum()

        if scaler is not None:
            scaled_loss = scaler.scale(loss)
            assert isinstance(scaled_loss, torch.Tensor)
            scaled_loss.backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
            optimizer.step()

        optimizer_steps += 1
        running_train_loss += loss.item()
        train_loss_steps += 1

    def _train_on_current_buffer() -> None:
        if buffer_training_mode == "full_batch":
            images, targets = training_buffer.get_batch()
            for _ in range(train_steps_per_buffer):
                _optimizer_step(images, targets)
        elif buffer_training_mode == "mini_batch":
            for _ in range(local_epochs_per_buffer):
                mini_batches = training_buffer.get_minibatches(
                    mini_batch_size,
                    shuffle=shuffle_buffer_each_epoch,
                )
                for images, targets in mini_batches:
                    _optimizer_step(images, targets)
        else:
            raise ValueError(f"Unknown buffer_training_mode: {buffer_training_mode}")

        training_buffer.clear()

    pbar = tqdm(stream, desc="Streaming", total=total_items) if progress_bar else None
    item_iter: Iterable[StreamItem] = pbar if pbar is not None else stream

    for stream_item in item_iter:
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
            forward_pass = filter_policy.requires_model_forward()
            if "score" in meta:
                filter_metric = "distribution_score"
                filter_score = float(meta["score"])
            elif "uncertainty" in meta:
                filter_metric = "uncertainty"
                filter_score = float(meta["uncertainty"])
            elif "grad_norm" in meta:
                filter_metric = "grad_norm"
                filter_score = float(meta["grad_norm"])
            elif "random_score" in meta:
                filter_metric = "random_score"
                filter_score = float(meta["random_score"])
            else:
                filter_metric = "none"
                filter_score = 0.0
            filter_threshold = float(meta["threshold"]) if "threshold" in meta else None

            metrics_logger.log_stream_item(action, forward_pass=forward_pass)
            metrics_logger.log_decision(
                global_idx=stream_item.metadata.get("global_idx", items_processed - 1),
                checkpoint_idx=1 + ((items_processed - 1) // metrics_logger.checkpoint_interval),
                frame_id=stream_item.metadata.get("frame_id", ""),
                action=action,
                filter_metric=filter_metric,
                filter_score=filter_score,
                filter_threshold=filter_threshold,
                categories=stream_item.categories,
                is_novel=(novelty_tracker.last_was_novel if novelty_tracker else False),
            )

        # Train when buffer is full
        if training_buffer.is_full():
            _train_on_current_buffer()

        # Checkpoint and evaluation
        if metrics_logger is not None and metrics_logger.should_checkpoint():
            checkpoint_idx += 1

            filter_stats = filter_policy.get_stats()
            buffer_stats = training_buffer.get_stats()
            novelty_stats = novelty_tracker.get_stats() if novelty_tracker else None
            checkpoint_loss = (running_train_loss / train_loss_steps) if train_loss_steps > 0 else None
            metrics_logger.log_checkpoint(
                checkpoint_idx,
                optimizer_steps,
                filter_stats,
                buffer_stats,
                novelty_stats,
                avg_train_loss=checkpoint_loss,
            )
            running_train_loss = 0.0
            train_loss_steps = 0

            selection_stats = filter_policy.get_selection_stats()
            metrics_logger.log_filter_stats(checkpoint_idx, selection_stats)
            filter_policy.reset_selection_stats()

            if eval_fn is not None and checkpoint_idx % eval_every_n_checkpoints == 0:
                eval_metrics = eval_fn(model)
                metrics_logger.log_evaluation(checkpoint_idx, eval_metrics)

                if pbar is not None:
                    pbar.set_postfix({
                        "mAP": f"{eval_metrics.get('mAP', 0.0):.3f}",
                        "accept": f"{filter_stats.get('accept_rate', 1.0):.2f}",
                    })

    # Final partial-buffer flush to avoid dropping accepted tail items.
    if len(training_buffer) > 0:
        _train_on_current_buffer()

    return StreamingTrainResult(
        items_processed=items_processed,
        items_accepted=items_accepted,
        items_rejected=items_rejected,
        buffer_flushes=training_buffer.total_flushes,
        optimizer_steps=optimizer_steps,
    )
