"""Streaming detection experiment.

Two-phase pipeline:
    Phase 1 (Bootstrap): multi-epoch training on the first N frames.
    Also collects bootstrap embedding statistics (for the distribution
    filter) or bootstrap uncertainty scores (for the uncertainty filter).

    Phase 2 (Streaming): single-pass, buffer-based training over the
    remaining frames with active filtering.  Supports a static or
    adaptive filter; the adaptive mode periodically snapshots the live
    model into the scoring model and recomputes the reference over the
    bootstrap frames plus a window / reservoir of recent accepts.

Bootstrap reuse: set bootstrap_run_dir in the config (or
--bootstrap-run-dir on the CLI) to skip Phase 1 and load the model,
embeddings, and/or uncertainty scores from a previous run.  This saves
hours when comparing filter policies on the same bootstrap.

Usage:
    python experiments/streaming.py \\
        --config configs/streaming/no_filter_cityday_curated.yaml
    python experiments/streaming.py \\
        --config configs/streaming/adaptive_reservoir_p15_cityday_curated.yaml \\
        --bootstrap-run-dir outputs/streaming/no_filter_cityday_curated/seed_42/<timestamp>
"""

from __future__ import annotations

import argparse
import copy
import csv
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
import torch.nn as nn
import yaml

warnings.filterwarnings("ignore", message="Can't initialize NVML")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from stream_active_fl.core import (
    DetectionDataset,
    DetectionStream,
    build_class_mapping,
    detection_collate,
    get_detection_augmentation,
    get_detection_transforms,
    load_manifest,
)
from stream_active_fl.evaluation import (
    DEFAULT_DOMAIN_DIMS,
    EXTENDED_DOMAIN_DIMS,
    attach_stream_blocks,
    evaluate_detection,
)
from stream_active_fl.runtime import (
    build_detector_from_config,
    load_dataclass_config,
    resolve_manifest_path,
    setup_run_dir,
)
from stream_active_fl.tracking import StreamingMetricsLogger, log_gpu_memory, save_run_info
from stream_active_fl.memory import TrainingBuffer
from stream_active_fl.policies import (
    DetectionUncertaintyPolicy,
    DistributionBasedPolicy,
    MixturePolicy,
    RefreshRecord,
    ScoringRefresher,
    create_filter_policy,
)
from stream_active_fl.training import (
    bootstrap_train,
    collect_embeddings,
    collect_uncertainties,
    train_on_stream,
)
from stream_active_fl.utils import set_seed, worker_init_fn


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StreamingDetectionConfig:
    """Configuration for the streaming detection experiment."""

    # Paths
    manifest_path: str = ""
    output_dir: str = "outputs/streaming/no_filter"

    # Model / classes
    num_classes: int = 11
    target_classes: Optional[List[str]] = None
    trainable_backbone_layers: int = 3
    image_min_size: int = 480
    image_max_size: int = 1600
    pretrained_backbone: bool = True
    pretrained_detector: bool = True
    load_checkpoint: Optional[str] = None

    # Bootstrap phase (skipped when bootstrap_run_dir is set)
    bootstrap_run_dir: Optional[str] = None
    bootstrap_frames: int = 5000
    bootstrap_epochs: int = 20
    bootstrap_batch_size: int = 8
    bootstrap_lr: float = 4e-4

    # Streaming phase -- training
    streaming_lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    buffer_capacity: int = 16
    train_steps_per_buffer: int = 1
    buffer_training_mode: Literal["full_batch", "mini_batch"] = "full_batch"
    local_epochs_per_buffer: int = 1
    mini_batch_size: int = 8
    shuffle_buffer_each_epoch: bool = True

    # Augmentation.  Applied to bootstrap training only; the bootstrap
    # embedding-collection pass and the streaming phase both use un-augmented
    # frames so the filter's scorer sees deterministic inputs.
    augment: bool = True
    hflip_prob: float = 0.5
    color_jitter: bool = True

    # Filtering policy
    # - "none":                accept all frames (no_filter baseline)
    # - "random":              accept each frame with probability accept_fraction
    # - "distribution":        Mahalanobis-distance filter with a
    #                          bootstrap-calibrated threshold (optionally
    #                          adaptive via scoring_refresh_*)
    # - "uncertainty":         top-K / margin detection-confidence uncertainty
    #                          filter (optionally adaptive via scoring_refresh_*)
    # - "mixed_distribution":  epsilon-greedy mixture of the distribution
    #                          filter and random (mixture_gamma routes signal)
    # - "mixed_uncertainty":   epsilon-greedy mixture of the uncertainty
    #                          filter and random (mixture_gamma routes signal)
    filter_policy: Literal[
        "none", "random", "distribution", "uncertainty",
        "mixed_distribution", "mixed_uncertainty",
    ] = "none"
    accept_fraction: float = 0.10

    # Threshold percentile: fraction of the reference distribution that should
    # fall at or above the threshold.  E.g. 0.15 calibrates the threshold at
    # the 85th percentile of reference scores; a stream frame is accepted iff
    # its score is at least that large.  Used by distribution, uncertainty,
    # and the mixed variants.
    threshold_percentile: float = 0.10

    # Uncertainty-policy: number of top-confidence detections averaged per
    # frame in topk_mean mode; reduction mode selects between top-K mean
    # confidence and top1 - top2 margin.
    uncertainty_top_k: int = 10
    uncertainty_score_mode: Literal["topk_mean", "margin"] = "topk_mean"

    # Mixture-policy routing: fraction of frames handled by the signal-based
    # inner policy; the remaining (1 - mixture_gamma) follow a random
    # accept_fraction coin.  Ignored unless filter_policy is a mixed variant.
    mixture_gamma: float = 0.5

    # Streaming LR schedule (linear warmup + cosine decay)
    streaming_lr_warmup_items: int = 0
    streaming_lr_min_factor: float = 0.1

    # Adaptive filter refresh.  Set scoring_refresh_every_flushes > 0 to
    # enable: every K buffer flushes the scoring model is replaced with a
    # snapshot of the live model and the reference (mean/cov/threshold) is
    # recomputed over the bootstrap frames plus the accepted-frame buffer.
    # The accepted buffer is controlled by exactly one of:
    #   scoring_refresh_window_size: deque of the last M accepted frames.
    #   scoring_refresh_reservoir_size: uniform random reservoir of size R
    #     over all past accepts (Vitter's Algorithm R).
    # Set both to 0 for a static reference (scoring model still refreshes,
    # but mean/cov/threshold are kept at the bootstrap values).
    scoring_refresh_every_flushes: int = 0
    scoring_refresh_window_size: int = 0
    scoring_refresh_reservoir_size: int = 0
    scoring_refresh_batch_size: int = 16
    # Reference distribution structure used by DistributionBasedPolicy:
    #   "single":         one Gaussian fitted to {bootstrap + accepted}
    #                     (legacy behavior).
    #   "two_reference":  two Gaussians, one on bootstrap and one on
    #                     accepted, with score = min(d_boot, d_adapt).
    #                     Avoids the unimodal-fit pathology when the
    #                     accepted set drifts away from the bootstrap.
    #                     Requires window_size > 0 or reservoir_size > 0.
    scoring_reference_mode: Literal["single", "two_reference"] = "single"
    # When False, the bootstrap is excluded from the reference at each
    # refresh and only the accepted window/reservoir is fitted (noBoot
    # ablation).  Default True preserves the legacy bootstrap-anchored
    # behavior.  Requires window > 0 or reservoir > 0; cannot combine
    # with scoring_reference_mode='two_reference'.
    scoring_include_bootstrap: bool = True

    # Evaluation
    eval_every_n_items: int = 5000
    checkpoint_interval: int = 1000
    score_threshold: float = 0.3
    min_box_area: float = 64.0
    bootstrap_smoke_check_frames: int = 200
    bootstrap_smoke_score_threshold: float = 0.3
    bootstrap_smoke_min_map50: float = 0.005
    bootstrap_fail_on_smoke_check: bool = True

    # Performance
    use_amp: bool = True

    # DataLoader
    num_workers: int = 2

    # Reproducibility
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StreamingDetectionConfig":
        return load_dataclass_config(cls, path)


# =============================================================================
# Main
# =============================================================================


def main(config: StreamingDetectionConfig, config_path: Path, command: str) -> None:
    start_time = datetime.now()

    print("=" * 60)
    print("Streaming Detection")
    print("=" * 60)

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    if config.use_amp:
        print("AMP: enabled (mixed-precision training)")

    manifest_path = resolve_manifest_path(PROJECT_ROOT, config.manifest_path)
    run_dir = setup_run_dir(PROJECT_ROOT, config.output_dir, config_path, seed=config.seed)
    print(f"Run directory: {run_dir}")

    class_mapping = build_class_mapping(config.target_classes)
    if config.target_classes is not None:
        config.num_classes = class_mapping.num_classes
        print(f"Target classes ({len(class_mapping.names)}): {', '.join(class_mapping.names)}")

    # Transforms + augmentation
    train_transform, val_transform = get_detection_transforms()
    train_augmentation = None
    if config.augment:
        train_augmentation = get_detection_augmentation(
            hflip_prob=config.hflip_prob,
            color_jitter=config.color_jitter,
        )

    # =========================================================================
    # Phase 1: Bootstrap (run or reuse)
    # =========================================================================

    requires_bootstrap_embeddings = config.filter_policy in (
        "distribution", "mixed_distribution",
    )
    requires_bootstrap_uncertainty = config.filter_policy in (
        "uncertainty", "mixed_uncertainty",
    )
    embedding_mean: Optional[torch.Tensor] = None
    embedding_cov: Optional[torch.Tensor] = None
    bootstrap_scores: Optional[torch.Tensor] = None

    # Resolve bootstrap_run_dir (config or CLI override)
    bootstrap_source: Optional[Path] = None
    if config.bootstrap_run_dir:
        p = Path(config.bootstrap_run_dir)
        bootstrap_source = p if p.is_absolute() else PROJECT_ROOT / p

    bootstrap_start = time.time()

    if bootstrap_source is not None:
        # ----- Reuse bootstrap from a previous run -----
        model_path = bootstrap_source / "bootstrap_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Bootstrap model not found: {model_path}")
        embed_path = bootstrap_source / "bootstrap_embeddings.pt"
        # If the source was a non-distribution run (e.g. no_filter), it will
        # not have bootstrap_embeddings.pt; we recompute them below from the
        # loaded model since they are a deterministic function of (weights,
        # bootstrap frames, transforms, target_classes, min_box_area).
        recompute_embeddings = (
            requires_bootstrap_embeddings and not embed_path.exists()
        )

        print("\n" + "=" * 60)
        print("Phase 1: Loading Bootstrap from Previous Run")
        print(f"  Source: {bootstrap_source}")
        print("=" * 60)

        # Warn if the source config used different bootstrap hyper-parameters
        source_config_path = bootstrap_source / "config.yaml"
        if source_config_path.exists():
            with open(source_config_path, "r") as f:
                source_cfg = yaml.safe_load(f)
            if isinstance(source_cfg, dict):
                for key in ("bootstrap_frames", "bootstrap_epochs", "bootstrap_lr",
                            "bootstrap_batch_size", "min_box_area", "trainable_backbone_layers"):
                    src_val = source_cfg.get(key)
                    cur_val = getattr(config, key, None)
                    if src_val is not None and cur_val is not None and src_val != cur_val:
                        print(f"  WARNING: {key} differs: source={src_val}, current={cur_val}")

        model = build_detector_from_config(config)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        print(model)
        print(f"Loaded bootstrap model from {model_path.name}")

        if requires_bootstrap_embeddings and not recompute_embeddings:
            embed_data = torch.load(embed_path, map_location="cpu", weights_only=True)
            embedding_mean = embed_data["mean"]
            embedding_cov = embed_data["cov"]
            if "scores" not in embed_data:
                raise KeyError(
                    "bootstrap_embeddings.pt is missing required key 'scores'. "
                    "Please regenerate bootstrap embeddings with the current code."
                )
            bootstrap_scores = embed_data["scores"]
            assert bootstrap_scores is not None
            assert embedding_mean is not None and embedding_cov is not None
            print(
                "Loaded embeddings:"
                f" mean {embedding_mean.shape}, cov {embedding_cov.shape},"
                f" scores {bootstrap_scores.shape}"
            )
        elif recompute_embeddings:
            print(
                f"  NOTE: {embed_path.name} not found at source; "
                f"recomputing embeddings from bootstrap_model.pt over the "
                f"first {config.bootstrap_frames} train frames (unaugmented)."
            )
            embed_dataset = DetectionDataset(
                manifest_path=manifest_path,
                split="train",
                transform=train_transform,
                augmentation=None,
                frame_range=(0, config.bootstrap_frames),
                min_box_area=config.min_box_area,
                target_classes=config.target_classes,
                verbose=False,
            )
            embed_loader = torch.utils.data.DataLoader(
                embed_dataset,
                batch_size=config.bootstrap_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=detection_collate,
                worker_init_fn=worker_init_fn,
                pin_memory=device.type == "cuda",
            )
            embedding_mean, embedding_cov, _embedding_count, bootstrap_scores = (
                collect_embeddings(model, embed_loader, device)
            )
            print(
                "Recomputed embeddings:"
                f" mean {embedding_mean.shape}, cov {embedding_cov.shape},"
                f" scores {bootstrap_scores.shape}"
            )

        # Copy artifacts into this run for provenance
        shutil.copy(model_path, run_dir / "bootstrap_model.pt")
        if requires_bootstrap_embeddings:
            if recompute_embeddings:
                torch.save(
                    {
                        "mean": embedding_mean,
                        "cov": embedding_cov,
                        "scores": bootstrap_scores,
                    },
                    run_dir / "bootstrap_embeddings.pt",
                )
            else:
                shutil.copy(embed_path, run_dir / "bootstrap_embeddings.pt")
        (run_dir / "bootstrap_source.txt").write_text(str(bootstrap_source))

    else:
        # ----- Run bootstrap from scratch -----
        print("\n" + "=" * 60)
        print("Phase 1: Bootstrap Training")
        print("=" * 60)

        bootstrap_dataset = DetectionDataset(
            manifest_path=manifest_path,
            split="train",
            transform=train_transform,
            augmentation=train_augmentation,
            frame_range=(0, config.bootstrap_frames),
            min_box_area=config.min_box_area,
            target_classes=config.target_classes,
        )

        bootstrap_loader = torch.utils.data.DataLoader(
            bootstrap_dataset,
            batch_size=config.bootstrap_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=detection_collate,
            worker_init_fn=worker_init_fn,
            pin_memory=device.type == "cuda",
        )

        model = build_detector_from_config(config)

        if config.load_checkpoint:
            checkpoint_path = PROJECT_ROOT / config.load_checkpoint
            print(f"Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])

        model = model.to(device)
        print(model)

        bootstrap_optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.bootstrap_lr,
            weight_decay=config.weight_decay,
        )

        print(f"\nBootstrap: {config.bootstrap_frames} frames, {config.bootstrap_epochs} epochs")
        epoch_logs, total_steps = bootstrap_train(
            model=model,
            train_loader=bootstrap_loader,
            optimizer=bootstrap_optimizer,
            device=device,
            epochs=config.bootstrap_epochs,
            max_grad_norm=config.max_grad_norm,
            scaler=scaler,
        )
        final_loss = epoch_logs[-1]["avg_loss"] if epoch_logs else float("nan")
        print(f"Bootstrap complete: final_loss={final_loss:.4f}, steps={total_steps}")

        with open(run_dir / "bootstrap_epochs.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "avg_loss", "batches"])
            writer.writeheader()
            writer.writerows(epoch_logs)

        print("\nRunning bootstrap smoke-check...")
        smoke_val_stream = DetectionStream(
            manifest_path=manifest_path,
            split="val",
            transform=val_transform,
            min_box_area=config.min_box_area,
            frame_range=(0, config.bootstrap_smoke_check_frames),
            target_classes=config.target_classes,
            verbose=False,
        )
        smoke_metrics = evaluate_detection(
            model,
            smoke_val_stream,
            device,
            score_threshold=config.bootstrap_smoke_score_threshold,
            use_amp=config.use_amp,
        )
        print(
            "Smoke-check:"
            f" mAP={smoke_metrics.get('mAP', 0.0):.4f},"
            f" mAP@50={smoke_metrics.get('mAP_50', 0.0):.4f},"
            f" preds={int(smoke_metrics.get('total_predictions', 0.0))},"
            f" gt={int(smoke_metrics.get('total_ground_truth', 0.0))},"
            f" items={int(smoke_metrics.get('num_items', 0.0))}"
        )
        if (
            config.bootstrap_fail_on_smoke_check
            and smoke_metrics.get("mAP_50", 0.0) < config.bootstrap_smoke_min_map50
        ):
            raise RuntimeError(
                "Bootstrap smoke-check failed: "
                f"mAP@50={smoke_metrics.get('mAP_50', 0.0):.4f} < "
                f"{config.bootstrap_smoke_min_map50:.4f}. "
                "Increase bootstrap strength or adjust detector initialization."
            )

        torch.save(
            {"model_state_dict": model.state_dict()},
            run_dir / "bootstrap_model.pt",
        )

        if requires_bootstrap_embeddings:
            # Collect embedding statistics from unaugmented images so the
            # reference distribution is deterministic and not jittered by
            # random color/flip augmentations.
            print("\nCollecting embedding statistics...")
            bootstrap_embed_dataset = DetectionDataset(
                manifest_path=manifest_path,
                split="train",
                transform=train_transform,
                augmentation=None,
                frame_range=(0, config.bootstrap_frames),
                min_box_area=config.min_box_area,
                target_classes=config.target_classes,
                verbose=False,
            )
            embed_loader = torch.utils.data.DataLoader(
                bootstrap_embed_dataset,
                batch_size=config.bootstrap_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=detection_collate,
                worker_init_fn=worker_init_fn,
                pin_memory=device.type == "cuda",
            )
            embedding_mean, embedding_cov, _embedding_count, bootstrap_scores = (
                collect_embeddings(model, embed_loader, device)
            )
            print(f"Embedding mean shape: {embedding_mean.shape}")
            print(f"Embedding cov shape:  {embedding_cov.shape}")
            print(f"Bootstrap scores:     {bootstrap_scores.shape} "
                  f"(min={bootstrap_scores.min():.3f}, median={bootstrap_scores.median():.3f}, "
                  f"max={bootstrap_scores.max():.3f})")

            torch.save(
                {
                    "mean": embedding_mean,
                    "cov": embedding_cov,
                    "scores": bootstrap_scores,
                },
                run_dir / "bootstrap_embeddings.pt",
            )

    # -------------------------------------------------------------------------
    # Bootstrap-uncertainty scores (for filter_policy == "uncertainty")
    # -------------------------------------------------------------------------
    # Compute (or reuse) per-frame detection-uncertainty scores over the
    # bootstrap frames using the bootstrap model.  These calibrate the
    # uncertainty threshold at threshold_percentile.  The model is not
    # re-trained here; we only run inference on unaugmented bootstrap frames.
    if requires_bootstrap_uncertainty:
        unc_artifact = run_dir / "bootstrap_uncertainties.pt"
        src_unc_path = (
            bootstrap_source / "bootstrap_uncertainties.pt"
            if bootstrap_source is not None
            else None
        )
        if src_unc_path is not None and src_unc_path.exists():
            unc_data = torch.load(src_unc_path, map_location="cpu", weights_only=True)
            # Cached scores are invalidated whenever top_k or score_mode
            # differs from the current config.  Legacy caches without a
            # score_mode key default to topk_mean for backwards compatibility.
            cached_top_k = unc_data.get("top_k") if isinstance(unc_data, dict) else None
            cached_mode = (
                unc_data.get("score_mode", "topk_mean")
                if isinstance(unc_data, dict)
                else None
            )
            if (
                isinstance(unc_data, dict)
                and cached_top_k == config.uncertainty_top_k
                and cached_mode == config.uncertainty_score_mode
                and "scores" in unc_data
            ):
                bootstrap_scores = unc_data["scores"]
                assert bootstrap_scores is not None
                print(
                    f"Loaded bootstrap uncertainties from {src_unc_path.name}: "
                    f"n={bootstrap_scores.numel()}, top_k={cached_top_k}, "
                    f"score_mode={cached_mode}"
                )
                shutil.copy(src_unc_path, unc_artifact)
            else:
                print(
                    f"  NOTE: {src_unc_path.name} found but does not match "
                    f"(top_k={config.uncertainty_top_k}, "
                    f"score_mode={config.uncertainty_score_mode}); "
                    "recomputing."
                )
                bootstrap_scores = None

        if bootstrap_scores is None:
            print("\nCollecting bootstrap uncertainty scores...")
            unc_dataset = DetectionDataset(
                manifest_path=manifest_path,
                split="train",
                transform=train_transform,
                augmentation=None,
                frame_range=(0, config.bootstrap_frames),
                min_box_area=config.min_box_area,
                target_classes=config.target_classes,
                verbose=False,
            )
            unc_loader = torch.utils.data.DataLoader(
                unc_dataset,
                batch_size=config.bootstrap_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=detection_collate,
                worker_init_fn=worker_init_fn,
                pin_memory=device.type == "cuda",
            )
            bootstrap_scores = collect_uncertainties(
                model, unc_loader, device,
                top_k=config.uncertainty_top_k,
                score_mode=config.uncertainty_score_mode,
            )
            print(
                f"Bootstrap uncertainties: n={bootstrap_scores.numel()} "
                f"(min={bootstrap_scores.min():.3f}, "
                f"median={bootstrap_scores.median():.3f}, "
                f"max={bootstrap_scores.max():.3f}, "
                f"top_k={config.uncertainty_top_k}, "
                f"score_mode={config.uncertainty_score_mode})"
            )
            torch.save(
                {
                    "scores": bootstrap_scores,
                    "top_k": config.uncertainty_top_k,
                    "score_mode": config.uncertainty_score_mode,
                },
                unc_artifact,
            )

    # =========================================================================
    # Phase 2: Streaming
    # =========================================================================

    bootstrap_duration = time.time() - bootstrap_start
    print(f"\nBootstrap phase completed in {bootstrap_duration:.1f}s")
    log_gpu_memory()

    print("\n" + "=" * 60)
    print("Phase 2: Streaming Training")
    print("=" * 60)

    # Stream starts after bootstrap frames.
    # No augmentation: the scorer sees raw camera frames (deterministic scores)
    # and the low-plasticity buffer regime (capacity=32, single pass) does not
    # benefit meaningfully from augmentation.
    train_stream = DetectionStream(
        manifest_path=manifest_path,
        split="train",
        transform=train_transform,
        augmentation=None,
        min_box_area=config.min_box_area,
        frame_range=(config.bootstrap_frames, None),
        target_classes=config.target_classes,
    )

    val_stream = DetectionStream(
        manifest_path=manifest_path,
        split="val",
        transform=val_transform,
        min_box_area=config.min_box_area,
        target_classes=config.target_classes,
    )

    # Streaming optimizer (fresh, not carrying bootstrap momentum)
    streaming_optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.streaming_lr,
        weight_decay=config.weight_decay,
    )

    # Frozen scoring model -- a snapshot of the bootstrap model used for
    # stable per-frame scoring.  Needed by every signal-based policy
    # (distribution, uncertainty, and the mixed_* variants).
    scoring_model: Optional[nn.Module] = None
    if config.filter_policy in (
        "distribution", "uncertainty",
        "mixed_distribution", "mixed_uncertainty",
    ):
        scoring_model = copy.deepcopy(model)
        scoring_model.eval()
        for p in scoring_model.parameters():
            p.requires_grad = False
        print("Created frozen scoring model for stable frame scoring")

    # Filter policy
    _bs_list = bootstrap_scores.tolist() if bootstrap_scores is not None else None
    filter_policy = create_filter_policy(
        config,
        bootstrap_mean=embedding_mean,
        bootstrap_cov=embedding_cov,
        scoring_model=scoring_model,
        bootstrap_scores=_bs_list,
    )
    print(f"Filter policy: {config.filter_policy}")

    # Adaptive filter refresh setup.  Active for any signal-based policy
    # (distribution, uncertainty, or mixed_*) whenever
    # scoring_refresh_every_flushes > 0.  For mixed policies the refresh
    # targets the inner signal-based policy; the MixturePolicy wrapper is
    # transparent to the refresher.
    refreshable_policy: Optional[
        "DistributionBasedPolicy | DetectionUncertaintyPolicy"
    ] = None
    if isinstance(filter_policy, (DistributionBasedPolicy, DetectionUncertaintyPolicy)):
        refreshable_policy = filter_policy
    elif isinstance(filter_policy, MixturePolicy) and isinstance(
        filter_policy.inner,
        (DistributionBasedPolicy, DetectionUncertaintyPolicy),
    ):
        refreshable_policy = filter_policy.inner

    adaptive_refresh_enabled = (
        config.filter_policy in (
            "distribution", "uncertainty",
            "mixed_distribution", "mixed_uncertainty",
        )
        and config.scoring_refresh_every_flushes > 0
        and refreshable_policy is not None
    )
    scoring_refresher: Optional[ScoringRefresher] = None
    refresh_records: List[RefreshRecord] = []
    on_refresh_cb: Optional[Callable[[int, int], None]] = None
    if adaptive_refresh_enabled:
        assert refreshable_policy is not None
        if (
            config.scoring_reference_mode == "two_reference"
            and config.scoring_refresh_window_size <= 0
            and config.scoring_refresh_reservoir_size <= 0
        ):
            raise ValueError(
                "scoring_reference_mode='two_reference' requires either "
                "scoring_refresh_window_size > 0 or "
                "scoring_refresh_reservoir_size > 0; without an accepted "
                "reference there is no second Gaussian to fit."
            )
        if (
            config.scoring_reference_mode == "two_reference"
            and not isinstance(refreshable_policy, DistributionBasedPolicy)
        ):
            raise ValueError(
                "scoring_reference_mode='two_reference' is only supported "
                "for DistributionBasedPolicy (filter_policy='distribution' "
                "or 'mixed_distribution')."
            )
        if not config.scoring_include_bootstrap:
            if (
                config.scoring_refresh_window_size <= 0
                and config.scoring_refresh_reservoir_size <= 0
            ):
                raise ValueError(
                    "scoring_include_bootstrap=False requires either "
                    "scoring_refresh_window_size > 0 or "
                    "scoring_refresh_reservoir_size > 0; without an "
                    "accepted reference there are no frames to fit at all."
                )
            if config.scoring_reference_mode == "two_reference":
                raise ValueError(
                    "scoring_include_bootstrap=False is incompatible with "
                    "scoring_reference_mode='two_reference': there is only "
                    "one set of frames (the accepted window/reservoir) so a "
                    "second Gaussian cannot be fitted."
                )
            if not isinstance(refreshable_policy, DistributionBasedPolicy):
                raise ValueError(
                    "scoring_include_bootstrap=False is only supported for "
                    "DistributionBasedPolicy."
                )
        manifest = load_manifest(manifest_path)
        all_train_entries = [f for f in manifest["frames"] if f["split"] == "train"]
        bootstrap_entries = all_train_entries[: config.bootstrap_frames]
        frame_id_to_entry = {f["frame_id"]: f for f in all_train_entries}
        scoring_refresher = ScoringRefresher(
            manifest_path=manifest_path,
            bootstrap_frame_entries=bootstrap_entries,
            frame_id_to_entry=frame_id_to_entry,
            transform=train_transform,
            target_classes=config.target_classes,
            min_box_area=config.min_box_area,
            batch_size=config.scoring_refresh_batch_size,
            num_workers=config.num_workers,
            device=device,
            use_amp=config.use_amp,
            reference_mode=config.scoring_reference_mode,
            include_bootstrap=config.scoring_include_bootstrap,
        )
        print(
            "Adaptive filter refresh enabled: every "
            f"{config.scoring_refresh_every_flushes} buffer flushes, "
            f"window={config.scoring_refresh_window_size}, "
            f"reservoir={config.scoring_refresh_reservoir_size}, "
            f"reference_mode={config.scoring_reference_mode}, "
            f"include_bootstrap={config.scoring_include_bootstrap}"
        )

        def _refresh_cb(items_processed: int, buffer_flushes: int) -> None:
            assert scoring_refresher is not None
            assert refreshable_policy is not None
            record = scoring_refresher.refresh(
                live_model=model,
                policies=[refreshable_policy],
                accepted_frame_ids=refreshable_policy.get_accepted_frame_ids(),
                trigger="buffer_flush",
                trigger_count=buffer_flushes,
            )
            refresh_records.append(record)
            print(
                f"    [refresh #{record.refresh_idx}] items={record.items_seen}, "
                f"flush={buffer_flushes}, window={record.window_size}, "
                f"ref={record.reference_size}, "
                f"thr: {record.threshold_before:.3f} -> {record.threshold_after:.3f}, "
                f"{record.duration_seconds:.1f}s"
            )
        on_refresh_cb = _refresh_cb

    # Training buffer
    training_buffer = TrainingBuffer(capacity=config.buffer_capacity)
    print(f"Training buffer capacity: {config.buffer_capacity}")

    # Metrics logger
    metrics_logger = StreamingMetricsLogger(
        log_dir=run_dir,
        checkpoint_interval=config.checkpoint_interval,
        class_names=class_mapping.names,
    )

    # Evaluation callback
    if config.checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be >= 1")
    eval_interval = max(1, config.eval_every_n_items // config.checkpoint_interval)

    # Per-frame val domain metadata keyed by frame_id so alignment
    # survives any frames skipped during iteration.  Marginal axes come
    # from the raw ZOD fields; a joint stream_block label is attached
    # when the manifest ordering strategy is recognized.
    val_domain_labels: Dict[str, Dict[str, Any]] = {
        str(f["frame_id"]): {
            dim: f.get(dim) for dim in DEFAULT_DOMAIN_DIMS
        }
        for f in val_stream.frames
    }
    manifest_for_blocks = load_manifest(manifest_path)
    ordering_strategy = (
        manifest_for_blocks.get("ordering", {}).get("strategy")
        if isinstance(manifest_for_blocks.get("ordering"), dict)
        else None
    )
    has_stream_block = attach_stream_blocks(
        val_domain_labels, list(val_stream.frames), ordering_strategy,
    )
    eval_domain_dims = list(EXTENDED_DOMAIN_DIMS) if has_stream_block else list(DEFAULT_DOMAIN_DIMS)

    def eval_fn(m: nn.Module) -> dict:
        return evaluate_detection(
            m,
            val_stream,
            device,
            score_threshold=config.score_threshold,
            use_amp=config.use_amp,
            domain_labels=val_domain_labels,
            domain_dims=eval_domain_dims,
        )

    dataset_info = {
        "bootstrap_frames": config.bootstrap_frames,
        "stream_frames": len(train_stream),
        "val_frames": len(val_stream),
        "bootstrap_reused": bootstrap_source is not None,
    }
    if bootstrap_source is not None:
        dataset_info["bootstrap_source"] = str(bootstrap_source)

    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        dataset_info=dataset_info,
        repo_path=PROJECT_ROOT,
    )

    # Run streaming training
    stream_length = len(train_stream)
    use_lr_schedule = config.streaming_lr_warmup_items > 0
    if use_lr_schedule:
        print(
            f"LR schedule: warmup {config.streaming_lr_warmup_items} items, "
            f"cosine decay to {config.streaming_lr * config.streaming_lr_min_factor:.2e}"
        )

    print(f"\nStreaming {stream_length} frames...")
    result = train_on_stream(
        model=model,
        stream=train_stream,
        optimizer=streaming_optimizer,
        filter_policy=filter_policy,
        training_buffer=training_buffer,
        device=device,
        max_grad_norm=config.max_grad_norm,
        train_steps_per_buffer=config.train_steps_per_buffer,
        buffer_training_mode=config.buffer_training_mode,
        local_epochs_per_buffer=config.local_epochs_per_buffer,
        mini_batch_size=config.mini_batch_size,
        shuffle_buffer_each_epoch=config.shuffle_buffer_each_epoch,
        metrics_logger=metrics_logger,
        eval_fn=eval_fn,
        eval_every_n_checkpoints=eval_interval,
        total_items=stream_length,
        scaler=scaler,
        base_lr=config.streaming_lr if use_lr_schedule else None,
        lr_warmup_items=config.streaming_lr_warmup_items,
        lr_min_factor=config.streaming_lr_min_factor,
        best_model_dir=run_dir,
        refresh_every_flushes=config.scoring_refresh_every_flushes if adaptive_refresh_enabled else 0,
        on_refresh=on_refresh_cb,
    )

    print("\n" + "=" * 60)
    print("Streaming complete!")
    print(f"  Items processed : {result.items_processed}")
    print(f"  Accepted        : {result.items_accepted}")
    print(f"  Rejected        : {result.items_rejected}")
    print(f"  Buffer flushes  : {result.buffer_flushes}")
    print(f"  Optimizer steps : {result.optimizer_steps}")
    if adaptive_refresh_enabled:
        print(f"  Refreshes       : {len(refresh_records)}")
    if result.best_eval_mAP > 0:
        print(f"  Best mAP        : {result.best_eval_mAP:.4f} (checkpoint {result.best_eval_checkpoint})")
    print("=" * 60)

    if adaptive_refresh_enabled and refresh_records:
        refresh_csv = run_dir / "refreshes.csv"
        with open(refresh_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "refresh_idx", "trigger", "trigger_count", "items_seen",
                    "window_size", "reference_size",
                    "threshold_before", "threshold_after", "duration_seconds",
                ],
            )
            writer.writeheader()
            for r in refresh_records:
                writer.writerow(r.__dict__)
        print(f"Refresh log: {refresh_csv}")

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_detection(
        model, val_stream, device, score_threshold=config.score_threshold,
        use_amp=config.use_amp,
        domain_labels=val_domain_labels,
        domain_dims=eval_domain_dims,
    )
    print(f"  mAP:    {final_metrics['mAP']:.4f}")
    print(f"  mAP@50: {final_metrics.get('mAP_50', 0.0):.4f}")
    print(f"  mAP@75: {final_metrics.get('mAP_75', 0.0):.4f}")

    metrics_logger.print_summary()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "final_metrics": final_metrics,
        },
        run_dir / "final_model.pt",
    )

    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        end_time=datetime.now(),
        best_metric=final_metrics["mAP"],
        metric_key="final_val_mAP",
        dataset_info=dataset_info,
        repo_path=PROJECT_ROOT,
        extra_info={
            "best_stream_mAP": result.best_eval_mAP,
            "best_stream_checkpoint": result.best_eval_checkpoint,
        },
    )

    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming detection experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--bootstrap-run-dir",
        type=str,
        default=None,
        help="Reuse bootstrap model + embeddings from this run directory "
             "(skips Phase 1). Overrides bootstrap_run_dir in config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from config. Run artefacts are placed under "
             "outputs/<exp>/seed_<N>/<timestamp>/ for multi-seed experiments.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    command = " ".join(sys.argv)
    config = StreamingDetectionConfig.from_yaml(config_path)

    if args.bootstrap_run_dir:
        config.bootstrap_run_dir = args.bootstrap_run_dir
    if args.seed is not None:
        config.seed = args.seed

    main(config, config_path, command)
