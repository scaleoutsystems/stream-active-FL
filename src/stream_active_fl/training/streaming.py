"""
Reusable streaming training loops for classification and detection.

Extracted from experiment scripts so the same training logic can be used
in centralized experiments and federated learning (where each client
runs the same loop on its own stream partition).

Key functions:
    train_on_classification_stream   Train a classifier on a stream of items
    train_on_detection_stream        Train a detector on a stream of items

Training utilities:
    RunningPosWeight                 Online pos_weight estimator for BCEWithLogitsLoss
    perform_classification_update    Single-item gradient step (classification)
    perform_detection_update         Single-item gradient step (detection)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ..core.items import StreamItem
from ..logging import StreamingMetricsLogger
from ..memory import ReplayBuffer
from ..policies import FilterPolicy


# =============================================================================
# Training result
# =============================================================================


@dataclass
class StreamingTrainResult:
    """Summary returned after a streaming training run.

    Useful for FL clients to report how much work they did (for
    weighted aggregation).
    """

    items_processed: int
    items_trained: int
    optimizer_steps: int


# =============================================================================
# Running pos_weight estimator
# =============================================================================


class RunningPosWeight:
    """
    Maintains a running estimate of pos_weight for BCEWithLogitsLoss.

    Updates incrementally as stream items are observed, so the loss weighting
    reflects only data seen so far (honest streaming -- no peeking at future
    class ratios). Starts at 1.0 (balanced assumption) and converges as more
    items arrive.

    The weight is ``n_negative / n_positive``, clamped to [0.1, 20.0] to
    avoid extreme values during early warm-up when counts are small.
    """

    def __init__(self, pos_weight_tensor: torch.Tensor):
        self._pos_weight_tensor = pos_weight_tensor
        self._n_positive = 0
        self._n_negative = 0

    def update(self, target: float) -> None:
        """Observe one stream item's binary target (0.0 or 1.0)."""
        if target >= 0.5:
            self._n_positive += 1
        else:
            self._n_negative += 1
        self._recompute()

    def _recompute(self) -> None:
        if self._n_positive == 0:
            weight = 1.0
        else:
            weight = self._n_negative / self._n_positive
        weight = max(0.1, min(weight, 20.0))
        self._pos_weight_tensor.fill_(weight)

    @property
    def value(self) -> float:
        return self._pos_weight_tensor.item()

    @property
    def n_positive(self) -> int:
        return self._n_positive

    @property
    def n_negative(self) -> int:
        return self._n_negative


# =============================================================================
# Single-item gradient updates
# =============================================================================


def perform_classification_update(
    model: nn.Module,
    criterion: nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    replay_batch: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    replay_weight: float = 0.5,
    accumulation_steps: int = 1,
) -> float:
    """
    Compute classification loss and accumulate gradients (backward pass only).

    The caller is responsible for ``optimizer.zero_grad()``, gradient clipping,
    and ``optimizer.step()``. This separation enables gradient accumulation
    across multiple stream items.

    Computes separate losses for the current item and replay batch, then
    combines them with explicit weighting. This prevents the replay batch
    (typically 32 samples) from drowning out the current item's gradient.

    Args:
        model: The classifier.
        criterion: Loss function (e.g. BCEWithLogitsLoss).
        image: Current stream item image tensor (C, H, W).
        target: Current stream item target tensor (scalar).
        replay_batch: Optional dict with ``"image"`` and ``"target"`` tensors.
        device: Device to run on.
        replay_weight: Weight for replay loss. Current item gets
            ``1 - replay_weight``. Default 0.5 gives equal weight.
        accumulation_steps: Divides loss by this value so accumulated
            gradient magnitude matches a single-step update.

    Returns:
        Unscaled combined loss value (for logging).
    """
    model.train()

    image = image.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)
    logits_current = model(image)
    loss_current = criterion(logits_current, target)

    if replay_batch is not None:
        replay_images = replay_batch["image"].to(device)
        replay_targets = replay_batch["target"].to(device)
        logits_replay = model(replay_images)
        loss_replay = criterion(logits_replay, replay_targets)

        loss = (1.0 - replay_weight) * loss_current + replay_weight * loss_replay
    else:
        loss = loss_current

    (loss / accumulation_steps).backward()

    return loss.item()


def perform_detection_update(
    model: nn.Module,
    stream_item: StreamItem,
    replay_batch: Optional[Dict[str, Any]],
    device: torch.device,
    replay_weight: float = 0.5,
    accumulation_steps: int = 1,
) -> float:
    """
    Compute detection loss and accumulate gradients (backward pass only).

    The detection model computes its own loss internally (classification +
    bbox regression + centerness). Current item and replay batch get
    separate forward passes with explicit weighting.

    Args:
        model: Detection model (e.g. FCOS).
        stream_item: Current stream item with annotations.
        replay_batch: Optional dict with ``"images"`` and ``"targets"`` lists.
        device: Device to run on.
        replay_weight: Weight for replay loss. Current gets
            ``1 - replay_weight``.
        accumulation_steps: Divides loss by this value for accumulation.

    Returns:
        Unscaled combined loss value (for logging).
    """
    model.train()

    image = stream_item.image.to(device)
    target = {
        "boxes": stream_item.annotations["boxes"].to(device),
        "labels": stream_item.annotations["labels"].to(device),
    }
    loss_dict = model([image], [target])
    loss_current = sum(loss_dict.values())

    if replay_batch is not None:
        replay_images = [img.to(device) for img in replay_batch["images"]]
        replay_targets = [
            {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
            for t in replay_batch["targets"]
        ]
        loss_dict_replay = model(replay_images, replay_targets)
        loss_replay = sum(loss_dict_replay.values())

        loss = (1.0 - replay_weight) * loss_current + replay_weight * loss_replay
    else:
        loss = loss_current

    (loss / accumulation_steps).backward()

    return loss.item()


# =============================================================================
# Optimizer step helper
# =============================================================================


def _maybe_optimizer_step(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    max_grad_norm: float,
) -> None:
    """Clip gradients (if configured) and step the optimizer."""
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_grad_norm,
        )
    optimizer.step()
    optimizer.zero_grad()


# =============================================================================
# Classification streaming training loop
# =============================================================================


def train_on_classification_stream(
    model: nn.Module,
    stream: Iterable[StreamItem],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    filter_policy: FilterPolicy,
    device: torch.device,
    *,
    replay_buffer: Optional[ReplayBuffer] = None,
    replay_batch_size: int = 32,
    replay_weight: float = 0.5,
    max_grad_norm: float = 0.0,
    accumulation_steps: int = 1,
    max_items: Optional[int] = None,
    running_pos_weight: Optional[RunningPosWeight] = None,
    filter_computes_forward: bool = False,
    metrics_logger: Optional[StreamingMetricsLogger] = None,
    eval_fn: Optional[Callable[[nn.Module], Dict[str, Any]]] = None,
    eval_every_n_checkpoints: int = 1,
    progress_bar: bool = True,
    total_items: Optional[int] = None,
) -> StreamingTrainResult:
    """
    Train a classifier on a stream of items in temporal order.

    Processes stream items one at a time, applying the filter policy to decide
    whether to train on each item. Optionally uses replay, logs metrics, and
    evaluates periodically.

    This function is the core training loop shared by centralized streaming
    experiments and federated clients. When called without logging/eval
    arguments it performs pure training suitable for FL.

    Args:
        model: The classifier to train.
        stream: Iterable of :class:`StreamItem` (e.g. ``StreamingDataset``
            or a client's sub-stream iterator).
        criterion: Loss function (e.g. ``BCEWithLogitsLoss``).
        optimizer: Optimizer for the model parameters.
        filter_policy: Policy that decides train/store/skip per item.
        device: Device to run computations on.
        replay_buffer: Optional replay buffer. Items are always offered to
            the buffer; replay batches are sampled when training.
        replay_batch_size: Batch size for replay sampling.
        replay_weight: Weight for replay loss vs. current item loss.
        max_grad_norm: Maximum gradient norm for clipping (0 = disabled).
        accumulation_steps: Accumulate gradients over this many items
            before stepping the optimizer.
        max_items: Stop after processing this many items (``None`` = exhaust
            the stream). Useful for FL where each client processes a fixed
            budget per round.
        running_pos_weight: Optional online pos_weight estimator.
        filter_computes_forward: Set to ``True`` when the filter policy
            performs its own forward pass (e.g. difficulty-based). Used
            for accurate forward-pass counting in the metrics logger.
        metrics_logger: Optional logger for streaming metrics and checkpoints.
        eval_fn: Optional evaluation callback ``(model) -> metrics_dict``,
            called at checkpoint intervals.
        eval_every_n_checkpoints: Evaluate every N checkpoints (default 1).
        progress_bar: Show a tqdm progress bar.
        total_items: Total expected items (for progress bar). Automatically
            used from ``stream`` if it has ``__len__``.

    Returns:
        :class:`StreamingTrainResult` with processing statistics.
    """
    if total_items is None and hasattr(stream, "__len__"):
        total_items = len(stream)

    checkpoint_idx = 0
    items_processed = 0
    train_count = 0
    optimizer_steps = 0

    optimizer.zero_grad()

    pbar = tqdm(stream, desc="Processing stream", total=total_items) if progress_bar else stream

    for stream_item in pbar:
        if max_items is not None and items_processed >= max_items:
            break

        items_processed += 1

        if running_pos_weight is not None:
            running_pos_weight.update(stream_item.target)

        action = filter_policy.select_action(stream_item, model, criterion, device)

        if metrics_logger is not None:
            forward_pass = (action == "train") or filter_computes_forward
            metrics_logger.log_stream_item(action, forward_pass, backward_pass=(action == "train"))

        if action == "train":
            replay_batch = None
            if replay_buffer is not None and len(replay_buffer) > 0:
                replay_batch = replay_buffer.sample(replay_batch_size, device=str(device))

            perform_classification_update(
                model,
                criterion,
                stream_item.image,
                torch.tensor(stream_item.target, dtype=torch.float32),
                replay_batch,
                device,
                replay_weight=replay_weight,
                accumulation_steps=accumulation_steps,
            )

            train_count += 1

            if train_count % accumulation_steps == 0:
                _maybe_optimizer_step(optimizer, model, max_grad_norm)
                optimizer_steps += 1

        if replay_buffer is not None:
            replay_buffer.add(stream_item.to_dict())

        # Checkpoint and evaluation
        if metrics_logger is not None and metrics_logger.should_checkpoint():
            checkpoint_idx += 1

            buffer_stats = replay_buffer.get_stats() if replay_buffer else None
            filter_stats = filter_policy.get_stats()
            metrics_logger.log_checkpoint(checkpoint_idx, buffer_stats, filter_stats)

            selection_stats = filter_policy.get_selection_stats()
            metrics_logger.log_filter_stats(checkpoint_idx, selection_stats)
            filter_policy.reset_selection_stats()

            if eval_fn is not None and checkpoint_idx % eval_every_n_checkpoints == 0:
                eval_metrics = eval_fn(model)
                metrics_logger.log_evaluation(checkpoint_idx, eval_metrics)

                if progress_bar and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({
                        "val_f1": f"{eval_metrics.get('f1', 0.0):.3f}",
                        "train_rate": f"{filter_stats.get('train_rate', 1.0):.3f}",
                    })

    # Flush remaining accumulated gradients
    if train_count % accumulation_steps != 0:
        _maybe_optimizer_step(optimizer, model, max_grad_norm)
        optimizer_steps += 1

    if running_pos_weight is not None:
        print(f"\nFinal running pos_weight: {running_pos_weight.value:.3f} "
              f"(pos={running_pos_weight.n_positive}, neg={running_pos_weight.n_negative})")

    return StreamingTrainResult(
        items_processed=items_processed,
        items_trained=train_count,
        optimizer_steps=optimizer_steps,
    )


# =============================================================================
# Detection streaming training loop
# =============================================================================


def train_on_detection_stream(
    model: nn.Module,
    stream: Iterable[StreamItem],
    optimizer: torch.optim.Optimizer,
    filter_policy: FilterPolicy,
    device: torch.device,
    *,
    replay_buffer: Optional[ReplayBuffer] = None,
    replay_batch_size: int = 16,
    replay_weight: float = 0.5,
    max_grad_norm: float = 0.0,
    accumulation_steps: int = 1,
    max_items: Optional[int] = None,
    filter_computes_forward: bool = False,
    metrics_logger: Optional[StreamingMetricsLogger] = None,
    eval_fn: Optional[Callable[[nn.Module], Dict[str, Any]]] = None,
    eval_every_n_checkpoints: int = 1,
    progress_bar: bool = True,
    total_items: Optional[int] = None,
) -> StreamingTrainResult:
    """
    Train a detector on a stream of items in temporal order.

    Same structure as :func:`train_on_classification_stream` but uses
    the detection-specific update path (model computes its own losses).

    Args:
        model: The detection model to train (e.g. FCOS).
        stream: Iterable of :class:`StreamItem` with detection annotations.
        optimizer: Optimizer for the model parameters.
        filter_policy: Policy that decides train/store/skip per item.
        device: Device to run computations on.
        replay_buffer: Optional replay buffer.
        replay_batch_size: Batch size for replay sampling.
        replay_weight: Weight for replay loss vs. current item loss.
        max_grad_norm: Maximum gradient norm for clipping (0 = disabled).
        accumulation_steps: Accumulate gradients over this many items.
        max_items: Stop after this many items (``None`` = exhaust stream).
        filter_computes_forward: Whether the filter does its own forward pass.
        metrics_logger: Optional logger for streaming metrics.
        eval_fn: Optional evaluation callback ``(model) -> metrics_dict``.
        eval_every_n_checkpoints: Evaluate every N checkpoints.
        progress_bar: Show a tqdm progress bar.
        total_items: Total expected items (for progress bar).

    Returns:
        :class:`StreamingTrainResult` with processing statistics.
    """
    if total_items is None and hasattr(stream, "__len__"):
        total_items = len(stream)

    # Detection models compute their own loss; filter policies accept
    # criterion=None and dispatch internally based on stream_item.annotations.
    criterion = None

    checkpoint_idx = 0
    items_processed = 0
    train_count = 0
    optimizer_steps = 0

    optimizer.zero_grad()

    pbar = tqdm(stream, desc="Processing stream", total=total_items) if progress_bar else stream

    for stream_item in pbar:
        if max_items is not None and items_processed >= max_items:
            break

        items_processed += 1

        action = filter_policy.select_action(stream_item, model, criterion, device)

        if metrics_logger is not None:
            forward_pass = (action == "train") or filter_computes_forward
            metrics_logger.log_stream_item(action, forward_pass, backward_pass=(action == "train"))

        if action == "train":
            replay_batch = None
            if replay_buffer is not None and len(replay_buffer) > 0:
                replay_batch = replay_buffer.sample(replay_batch_size, device=str(device))

            perform_detection_update(
                model,
                stream_item,
                replay_batch,
                device,
                replay_weight=replay_weight,
                accumulation_steps=accumulation_steps,
            )

            train_count += 1

            if train_count % accumulation_steps == 0:
                _maybe_optimizer_step(optimizer, model, max_grad_norm)
                optimizer_steps += 1

        if replay_buffer is not None:
            replay_buffer.add(stream_item.to_dict())

        # Checkpoint and evaluation
        if metrics_logger is not None and metrics_logger.should_checkpoint():
            checkpoint_idx += 1

            buffer_stats = replay_buffer.get_stats() if replay_buffer else None
            filter_stats = filter_policy.get_stats()
            metrics_logger.log_checkpoint(checkpoint_idx, buffer_stats, filter_stats)

            selection_stats = filter_policy.get_selection_stats()
            metrics_logger.log_filter_stats(checkpoint_idx, selection_stats)
            filter_policy.reset_selection_stats()

            if eval_fn is not None and checkpoint_idx % eval_every_n_checkpoints == 0:
                eval_metrics = eval_fn(model)
                metrics_logger.log_evaluation(checkpoint_idx, eval_metrics)

                if progress_bar and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({
                        "mAP": f"{eval_metrics.get('mAP', 0.0):.3f}",
                        "mAP50": f"{eval_metrics.get('mAP_50', 0.0):.3f}",
                        "train_rate": f"{filter_stats.get('train_rate', 1.0):.3f}",
                    })

    # Flush remaining accumulated gradients
    if train_count % accumulation_steps != 0:
        _maybe_optimizer_step(optimizer, model, max_grad_norm)
        optimizer_steps += 1

    return StreamingTrainResult(
        items_processed=items_processed,
        items_trained=train_count,
        optimizer_steps=optimizer_steps,
    )
