"""
Training loops for the two-phase streaming detection pipeline.

Phase 1 -- Bootstrap:
    bootstrap_train()          Multi-epoch training on the first N frames
    collect_embeddings()       Extract backbone embeddings for bootstrap frames
    collect_uncertainties()    Extract per-frame detection uncertainty scores

Phase 2 -- Streaming:
    train_on_stream()          Single-pass buffer-based streaming training
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sized,
    Tuple,
    overload,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..core.items import StreamItem
from ..tracking import StreamingMetricsLogger
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
    best_eval_mAP: float = 0.0
    best_eval_checkpoint: int = 0


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

        if epochs > 1:
            desc = f"{desc_prefix} epoch {epoch + 1}/{epochs}"
        else:
            desc = desc_prefix
        pbar = tqdm(train_loader, desc=desc) if progress_bar else None
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
        if epochs > 1:
            print(f"  Epoch {epoch + 1}/{epochs} -- avg loss: {avg_loss:.4f}")
        else:
            print(f"  {desc_prefix} -- avg loss: {avg_loss:.4f}")

    return epoch_logs, total_steps


@torch.no_grad()
@overload
def collect_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    progress_bar: bool = ...,
    use_amp: bool = ...,
    *,
    return_embeddings: Literal[False] = ...,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]: ...


@overload
def collect_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    progress_bar: bool = ...,
    use_amp: bool = ...,
    *,
    return_embeddings: Literal[True],
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]: ...


def collect_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    progress_bar: bool = True,
    use_amp: bool = False,
    return_embeddings: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor] | Tuple[
    torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor
]:
    """
    Collect backbone embeddings for all frames in a DataLoader.

    Runs a forward pass through each batch, extracting embeddings via
    model.get_embedding(). Computes the mean, covariance, and per-frame
    Mahalanobis distances of the embedding distribution.

    Args:
        model: The detection model with a get_embedding() method.
        data_loader: DataLoader over the frames to embed.
        device: Device to run on.
        progress_bar: Show progress bar.
        use_amp: Run embedding forward pass under torch.cuda.amp.autocast
            to speed up refresh; embeddings are cast back to float32 for
            numerically stable mean/covariance/Mahalanobis computation.
        return_embeddings: If True, also return the raw (N, D) embedding
            tensor.  Used by the two-reference refresh path which needs
            per-frame distances against a second Gaussian fitted on a
            different frame set.

    Returns:
        (mean, cov, count, scores) or, when return_embeddings=True,
        (mean, cov, count, scores, embeddings).  scores are the per-frame
        Mahalanobis distances of the input frames against the
        (mean, cov) Gaussian fitted on those same frames.
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
        with torch.cuda.amp.autocast(enabled=use_amp):
            emb = model.get_embedding(images)  # (B, D)
        all_embeddings.append(emb.float().cpu())

    if was_training:
        model.train()

    embeddings = torch.cat(all_embeddings, dim=0).float()  # (N, D)

    mean = embeddings.mean(dim=0)
    centered = embeddings - mean.unsqueeze(0)
    cov = (centered.T @ centered) / max(len(embeddings) - 1, 1)

    reg = 1e-5 * torch.eye(cov.shape[0])
    cov_inv = torch.linalg.inv(cov + reg)
    scores = torch.sqrt(
        (centered @ cov_inv * centered).sum(dim=1)
    )  # (N,) Mahalanobis distances

    if return_embeddings:
        return mean, cov, int(len(embeddings)), scores, embeddings
    return mean, cov, int(len(embeddings)), scores


def mahalanobis_distances(
    embeddings: torch.Tensor,
    mean: torch.Tensor,
    cov: torch.Tensor,
    *,
    cov_inv: Optional[torch.Tensor] = None,
    reg_eps: float = 1e-5,
) -> torch.Tensor:
    """Per-row Mahalanobis distance of `embeddings` (N, D) to (mean, cov).

    Args:
        embeddings: (N, D) float tensor.
        mean: (D,) float tensor.
        cov: (D, D) float tensor.
        cov_inv: Optional precomputed (D, D) inverse covariance.  Avoids
            a redundant inv when the caller already has it.
        reg_eps: Regularization added to the covariance diagonal before
            inversion for numerical stability.  Ignored when cov_inv is
            provided.

    Returns:
        (N,) tensor of Mahalanobis distances, all on CPU/float32.
    """
    emb = embeddings.float()
    mu = mean.float()
    if cov_inv is None:
        sigma = cov.float()
        reg = reg_eps * torch.eye(sigma.shape[0], dtype=sigma.dtype, device=sigma.device)
        cov_inv = torch.linalg.inv(sigma + reg)
    centered = emb - mu.unsqueeze(0)
    return torch.sqrt((centered @ cov_inv * centered).sum(dim=1))


@torch.no_grad()
def _frame_uncertainty_score(
    scores: Optional[torch.Tensor], top_k: int, score_mode: str,
) -> float:
    """Single-frame uncertainty score from post-NMS box confidences.

    Args:
        scores: Per-box classification confidences, sorted descending, or
            None for a frame with no detections.
        top_k: Number of highest-confidence boxes used by the topk_mean
            mode.  Ignored by the margin mode.
        score_mode: Either "topk_mean" (uncertainty = 1 - mean of top-K
            confidences) or "margin" (uncertainty = 1 - (top1 - top2);
            reduces to 1 - top1 when only one detection is present).

    Returns:
        Float uncertainty score in [0, 1].  Frames with no detections
        score 1.0 regardless of mode.
    """
    if scores is None or scores.numel() == 0:
        return 1.0
    s = scores.detach().float().cpu()

    if score_mode == "topk_mean":
        top = s[:top_k]
        return float(max(0.0, min(1.0, 1.0 - top.mean().item())))

    if score_mode == "margin":
        top1 = float(s[0].item())
        top2 = float(s[1].item()) if s.numel() >= 2 else 0.0
        return float(max(0.0, min(1.0, 1.0 - (top1 - top2))))

    raise ValueError(
        f"Unknown score_mode: {score_mode!r}; expected 'topk_mean' or 'margin'."
    )


def collect_uncertainties(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    top_k: int = 10,
    score_mode: str = "topk_mean",
    progress_bar: bool = True,
) -> torch.Tensor:
    """
    Collect per-frame detection-uncertainty scores for all frames in a loader.

    Runs the detector in eval mode and reduces each frame's post-NMS
    classification confidences to a scalar uncertainty.  Two reductions
    are supported, selected by score_mode:

        topk_mean: score = 1 - mean(top_k box scores).  Low mean
        confidence -> high uncertainty.

        margin: score = 1 - (top1 - top2 box scores).  Two near-equal
        top detections -> high uncertainty.  Falls back to 1 - top1 when
        the frame has a single detection.

    Scores are clamped to [0, 1].  Frames with zero detections score 1.0.

    Args:
        model: The detection model (eval mode enforced internally).
        data_loader: DataLoader over the frames to score.
        device: Device to run on.
        top_k: Number of top-confidence detections used by topk_mean.
        score_mode: "topk_mean" or "margin".  See above.
        progress_bar: Show progress bar.

    Returns:
        Tensor of shape (N,) with per-frame uncertainty scores in [0, 1].
    """
    was_training = model.training
    model.eval()

    scores: List[float] = []
    iterator = (
        tqdm(data_loader, desc="Collecting uncertainties")
        if progress_bar
        else data_loader
    )

    for batch in iterator:
        if batch is None:
            continue
        images, _ = batch
        images = [img.to(device) for img in images]
        preds = model(images)
        for p in preds:
            s = p.get("scores") if isinstance(p, dict) else None
            scores.append(_frame_uncertainty_score(s, top_k, score_mode))

    if was_training:
        model.train()

    return torch.tensor(scores, dtype=torch.float32)


# =============================================================================
# Phase 2: Buffer-based streaming training
# =============================================================================


def _compute_streaming_lr(
    items_processed: int,
    base_lr: float,
    warmup_items: int,
    total_items: int,
    min_factor: float,
) -> float:
    """Linear warmup followed by cosine decay to base_lr * min_factor."""
    if items_processed < warmup_items:
        return base_lr * (items_processed / max(warmup_items, 1))
    progress = (items_processed - warmup_items) / max(total_items - warmup_items, 1)
    progress = min(progress, 1.0)
    return base_lr * (min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


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
    progress_bar: bool = True,
    total_items: Optional[int] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    base_lr: Optional[float] = None,
    lr_warmup_items: int = 0,
    lr_min_factor: float = 0.1,
    best_model_dir: Optional[Path] = None,
    refresh_every_flushes: int = 0,
    on_refresh: Optional[Callable[[int, int], None]] = None,
    decision_callback: Optional[
        Callable[[int, str, str, str, float, Optional[float], Any], None]
    ] = None,
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
        progress_bar: Show progress bar.
        total_items: Total expected items (for progress bar).
        scaler: Optional GradScaler for AMP.
        base_lr: When set, enables linear warmup + cosine decay LR schedule.
        lr_warmup_items: Items for linear LR warmup (requires base_lr).
        lr_min_factor: Final LR = base_lr * lr_min_factor (requires base_lr).
        best_model_dir: When set, saves best_model.pt on each new best mAP.
        refresh_every_flushes: When > 0, call `on_refresh` after every K-th
            buffer flush.  Ignored when on_refresh is None.
        on_refresh: Callback invoked as on_refresh(items_processed,
            buffer_flushes) after a flush that triggers a refresh.  The
            callback owns the scoring-model + reference update.
        decision_callback: Optional per-item hook for recording filter
            decisions in addition to (or instead of) metrics_logger.
            Called as decision_callback(global_idx, frame_id, action,
            filter_metric, filter_score, filter_threshold, categories)
            for every stream item.  Used by the federated pipeline to
            write a shared decisions.csv across clients and rounds.

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

    use_lr_schedule = base_lr is not None and total_items is not None and total_items > 0

    best_eval_mAP = 0.0
    best_eval_checkpoint = 0

    running_train_loss = 0.0
    train_loss_steps = 0

    def _set_lr(lr: float) -> None:
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    def _optimizer_step(
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        nonlocal optimizer_steps, running_train_loss, train_loss_steps
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model.train()

        if use_lr_schedule:
            assert base_lr is not None and total_items is not None
            new_lr = _compute_streaming_lr(
                items_processed, base_lr, lr_warmup_items, total_items, lr_min_factor,
            )
            _set_lr(new_lr)

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

        if action == "accept":
            items_accepted += 1
            training_buffer.add(stream_item)
        else:
            items_rejected += 1

        # Derive filter decision metadata once; consumed by metrics_logger
        # and/or decision_callback below.  The policy can declare its metric
        # name via meta["metric"]; otherwise fall back to the legacy
        # heuristic (distribution vs random vs none).
        if metrics_logger is not None or decision_callback is not None:
            if "metric" in meta and "score" in meta:
                filter_metric = str(meta["metric"])
                filter_score = float(meta["score"])
            elif "score" in meta:
                filter_metric = "distribution_score"
                filter_score = float(meta["score"])
            elif "random_score" in meta:
                filter_metric = "random_score"
                filter_score = float(meta["random_score"])
            else:
                filter_metric = "none"
                filter_score = 0.0
            filter_threshold = (
                float(meta["threshold"]) if "threshold" in meta else None
            )
            global_idx = stream_item.metadata.get("global_idx", items_processed - 1)
            frame_id = stream_item.metadata.get("frame_id", "")

            if metrics_logger is not None:
                forward_pass = filter_policy.requires_model_forward()
                metrics_logger.log_stream_item(action, forward_pass=forward_pass)
                metrics_logger.log_decision(
                    global_idx=global_idx,
                    checkpoint_idx=1
                    + ((items_processed - 1) // metrics_logger.checkpoint_interval),
                    frame_id=frame_id,
                    action=action,
                    filter_metric=filter_metric,
                    filter_score=filter_score,
                    filter_threshold=filter_threshold,
                    categories=stream_item.categories,
                )

            if decision_callback is not None:
                decision_callback(
                    global_idx,
                    frame_id,
                    action,
                    filter_metric,
                    filter_score,
                    filter_threshold,
                    stream_item.categories,
                )

        # Train when buffer is full
        if training_buffer.is_full():
            _train_on_current_buffer()

            # Adaptive filter refresh: rebuild the scoring model and
            # reference distribution after every K-th buffer flush so the
            # notion of "in-distribution" follows the model and the
            # recent accept history.
            if (
                on_refresh is not None
                and refresh_every_flushes > 0
                and training_buffer.total_flushes % refresh_every_flushes == 0
            ):
                on_refresh(items_processed, training_buffer.total_flushes)

        # Checkpoint and evaluation
        if metrics_logger is not None and metrics_logger.should_checkpoint():
            checkpoint_idx += 1

            filter_stats = filter_policy.get_stats()
            buffer_stats = training_buffer.get_stats()
            checkpoint_loss = (running_train_loss / train_loss_steps) if train_loss_steps > 0 else None
            metrics_logger.log_checkpoint(
                checkpoint_idx,
                optimizer_steps,
                filter_stats,
                buffer_stats,
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

                current_mAP = eval_metrics.get("mAP", 0.0)
                if current_mAP > best_eval_mAP:
                    best_eval_mAP = current_mAP
                    best_eval_checkpoint = checkpoint_idx
                    if best_model_dir is not None:
                        torch.save(
                            {"model_state_dict": model.state_dict()},
                            best_model_dir / "best_model.pt",
                        )

                if pbar is not None:
                    pbar.set_postfix({
                        "mAP": f"{current_mAP:.3f}",
                        "best": f"{best_eval_mAP:.3f}",
                        "accept": f"{filter_stats.get('accept_rate', 1.0):.2f}",
                    })

    # Final partial-buffer flush to avoid dropping accepted tail items.
    if len(training_buffer) > 0:
        _train_on_current_buffer()

    # End-of-stream: log any items since the last checkpoint and always
    # run a final evaluation so that the CSV files cover the full stream.
    if metrics_logger is not None:
        has_unlogged_items = (
            items_processed % metrics_logger.checkpoint_interval != 0
        )
        last_checkpoint_was_eval = (
            checkpoint_idx > 0
            and checkpoint_idx % eval_every_n_checkpoints == 0
        )

        if has_unlogged_items:
            checkpoint_idx += 1
            filter_stats = filter_policy.get_stats()
            buffer_stats = training_buffer.get_stats()
            checkpoint_loss = (
                (running_train_loss / train_loss_steps)
                if train_loss_steps > 0
                else None
            )
            metrics_logger.log_checkpoint(
                checkpoint_idx, optimizer_steps, filter_stats, buffer_stats,
                avg_train_loss=checkpoint_loss,
            )
            selection_stats = filter_policy.get_selection_stats()
            metrics_logger.log_filter_stats(checkpoint_idx, selection_stats)

        if eval_fn is not None and (
            has_unlogged_items or not last_checkpoint_was_eval
        ):
            eval_metrics = eval_fn(model)
            metrics_logger.log_evaluation(checkpoint_idx, eval_metrics)
            current_mAP = eval_metrics.get("mAP", 0.0)
            if current_mAP > best_eval_mAP:
                best_eval_mAP = current_mAP
                best_eval_checkpoint = checkpoint_idx
                if best_model_dir is not None:
                    torch.save(
                        {"model_state_dict": model.state_dict()},
                        best_model_dir / "best_model.pt",
                    )

    return StreamingTrainResult(
        items_processed=items_processed,
        items_accepted=items_accepted,
        items_rejected=items_rejected,
        buffer_flushes=training_buffer.total_flushes,
        optimizer_steps=optimizer_steps,
        best_eval_mAP=best_eval_mAP,
        best_eval_checkpoint=best_eval_checkpoint,
    )
