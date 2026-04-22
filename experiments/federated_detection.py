"""Simulated federated learning for streaming object detection.

Pipeline:
    Phase 1 (Bootstrap): optional shared bootstrap model on the first N
    frames; reusable across runs via bootstrap_run_dir.

    Phase 2 (Federated): partition the remaining stream across clients,
    run local streaming updates, aggregate with FedAvg each round.
    Supports static and adaptive distribution filters; adaptive mode
    periodically snapshots the post-aggregation global model into the
    scoring model and recomputes one fleet-wide reference over the
    bootstrap frames plus a fleet-wide sample of recent accepts.

Usage:
    python experiments/federated_detection.py \\
        --config configs/federated/fed_no_filter_cityday_road_type.yaml
    python experiments/federated_detection.py \\
        --config configs/federated/fed_adaptive_cityday_road_type_p15.yaml \\
        --bootstrap-run-dir outputs/federated/fed_no_filter_cityday_road_type/seed_42/<timestamp>
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn

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
    partition_frames,
    partition_frames_by_domain,
)
from stream_active_fl.evaluation import evaluate_detection
from stream_active_fl.experiment import (
    build_detector_from_config,
    load_dataclass_config,
    resolve_manifest_path,
    setup_run_dir,
)
from stream_active_fl.logging import (
    FederatedDecisionsLogger,
    FederatedMetricsLogger,
    log_gpu_memory,
    save_run_info,
)
from stream_active_fl.memory import TrainingBuffer
from stream_active_fl.policies import (
    DistributionBasedPolicy,
    RefreshRecord,
    ScoringRefresher,
    create_filter_policy,
    pool_recent_accepted,
)
from stream_active_fl.training import bootstrap_train, collect_embeddings, fedavg, train_on_stream
from stream_active_fl.utils import set_seed, worker_init_fn


@dataclass
class FederatedDetectionConfig:
    # Paths
    manifest_path: str = ""
    output_dir: str = "outputs/federated/no_filter"

    # Model / classes
    num_classes: int = 11
    target_classes: Optional[List[str]] = None
    trainable_backbone_layers: int = 3
    image_min_size: int = 480
    image_max_size: int = 1600
    pretrained_backbone: bool = True
    pretrained_detector: bool = True
    load_checkpoint: Optional[str] = None

    # Shared bootstrap (optional reuse)
    bootstrap_run_dir: Optional[str] = None
    bootstrap_frames: int = 5000
    bootstrap_epochs: int = 20
    bootstrap_batch_size: int = 8
    bootstrap_lr: float = 4e-4

    # Federated setup
    num_clients: int = 4
    num_rounds: int = 10
    local_items_per_round: int = 2000
    partition_strategy: Literal["contiguous", "uniform", "domain_aligned"] = "contiguous"
    # Required when partition_strategy is "domain_aligned".
    # Each inner list names the manifest blocks assigned to that client.
    # Length must equal num_clients.  Blocks in each group must be adjacent
    # in the manifest's block_order.
    domain_client_groups: Optional[List[List[str]]] = None

    # Local streaming training
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
    # embedding-collection pass and every client's streaming phase use
    # un-augmented frames so the (shared) filter scorer sees deterministic
    # inputs.
    augment: bool = True
    hflip_prob: float = 0.5
    color_jitter: bool = True

    # Filtering policy (per-client).  Shares definitions with streaming.
    # - "none":         accept every frame (no_filter baseline)
    # - "random":       accept each frame with probability accept_fraction
    # - "distribution": Mahalanobis-distance filter with bootstrap-calibrated
    #                   threshold (optionally adaptive via scoring_refresh_*)
    filter_policy: Literal["none", "random", "distribution"] = "none"
    accept_fraction: float = 0.10
    threshold_percentile: float = 0.10

    # Adaptive filter refresh.  Set scoring_refresh_every_rounds > 0 to
    # enable: every K rounds the scoring model is replaced with a snapshot
    # of the post-aggregation global model and the reference (mean/cov/
    # threshold) is recomputed over the bootstrap frames plus the last M
    # accepted stream frames pooled across clients.  A single scoring model /
    # reference / threshold is broadcast to every client's policy
    # ("server-issued novelty definition").  Defaults recover the static
    # frozen filter baseline.
    scoring_refresh_every_rounds: int = 0
    scoring_refresh_window_size: int = 0
    scoring_refresh_reservoir_size: int = 0
    scoring_refresh_batch_size: int = 16

    # Evaluation
    eval_every_n_rounds: int = 1
    score_threshold: float = 0.3
    min_box_area: float = 64.0
    val_frame_limit: Optional[int] = None
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
    def from_yaml(cls, path: str | Path) -> "FederatedDetectionConfig":
        return load_dataclass_config(cls, path)


def _bootstrap_or_reuse(
    config: FederatedDetectionConfig,
    run_dir: Path,
    manifest_path: Path,
    train_transform,
    val_transform,
    train_augmentation,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[nn.Module, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[Path]]:
    requires_bootstrap_embeddings = (config.filter_policy == "distribution")
    bootstrap_source: Optional[Path] = None
    embedding_mean: Optional[torch.Tensor] = None
    embedding_cov: Optional[torch.Tensor] = None
    bootstrap_scores: Optional[torch.Tensor] = None

    if config.bootstrap_run_dir:
        p = Path(config.bootstrap_run_dir)
        bootstrap_source = p if p.is_absolute() else PROJECT_ROOT / p

    if bootstrap_source is not None:
        model_path = bootstrap_source / "bootstrap_model.pt"
        embed_path = bootstrap_source / "bootstrap_embeddings.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Bootstrap model not found: {model_path}")
        # If the source was a non-distribution run (e.g. no_filter), it will
        # not have bootstrap_embeddings.pt; we recompute them below from the
        # loaded model since they are a deterministic function of (weights,
        # bootstrap frames, transforms, target_classes, min_box_area).
        recompute_embeddings = (
            requires_bootstrap_embeddings and not embed_path.exists()
        )

        print("\n" + "=" * 60)
        print("Phase 1: Loading Shared Bootstrap")
        print(f"  Source: {bootstrap_source}")
        print("=" * 60)

        model = build_detector_from_config(config)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        print(f"Loaded bootstrap model from {model_path.name}")

        if requires_bootstrap_embeddings and not recompute_embeddings:
            embed_data = torch.load(embed_path, map_location="cpu", weights_only=True)
            if "scores" not in embed_data:
                raise KeyError(
                    "bootstrap_embeddings.pt is missing required key 'scores'. "
                    "Regenerate bootstrap embeddings with current code."
                )
            embedding_mean = embed_data["mean"]
            embedding_cov = embed_data["cov"]
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
        return model, embedding_mean, embedding_cov, bootstrap_scores, bootstrap_source

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
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    bootstrap_optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.bootstrap_lr,
        weight_decay=config.weight_decay,
    )

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

    smoke_stream = DetectionStream(
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
        smoke_stream,
        device,
        score_threshold=config.bootstrap_smoke_score_threshold,
        use_amp=config.use_amp,
    )
    print(
        "Smoke-check:"
        f" mAP={smoke_metrics.get('mAP', 0.0):.4f},"
        f" mAP@50={smoke_metrics.get('mAP_50', 0.0):.4f}"
    )
    if (
        config.bootstrap_fail_on_smoke_check
        and smoke_metrics.get("mAP_50", 0.0) < config.bootstrap_smoke_min_map50
    ):
        raise RuntimeError(
            "Bootstrap smoke-check failed: "
            f"mAP@50={smoke_metrics.get('mAP_50', 0.0):.4f} < "
            f"{config.bootstrap_smoke_min_map50:.4f}"
        )

    torch.save({"model_state_dict": model.state_dict()}, run_dir / "bootstrap_model.pt")

    if requires_bootstrap_embeddings:
        # Collect embedding statistics from unaugmented images so the
        # reference distribution is deterministic.
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
        torch.save(
            {
                "mean": embedding_mean,
                "cov": embedding_cov,
                "scores": bootstrap_scores,
            },
            run_dir / "bootstrap_embeddings.pt",
        )
        print(
            f"Embedding stats collected: mean {embedding_mean.shape}, "
            f"cov {embedding_cov.shape}, scores {bootstrap_scores.shape}"
        )

    return model, embedding_mean, embedding_cov, bootstrap_scores, None


def main(config: FederatedDetectionConfig, config_path: Path, command: str) -> None:
    start_time = datetime.now()
    print("=" * 60)
    print("Federated Streaming Detection")
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

    train_transform, val_transform = get_detection_transforms()
    train_augmentation = None
    if config.augment:
        train_augmentation = get_detection_augmentation(
            hflip_prob=config.hflip_prob,
            color_jitter=config.color_jitter,
        )

    bootstrap_start = time.time()
    global_model, embedding_mean, embedding_cov, bootstrap_scores, bootstrap_source = _bootstrap_or_reuse(
        config=config,
        run_dir=run_dir,
        manifest_path=manifest_path,
        train_transform=train_transform,
        val_transform=val_transform,
        train_augmentation=train_augmentation,
        device=device,
        scaler=scaler,
    )
    bootstrap_duration = time.time() - bootstrap_start
    print(f"\nBootstrap phase completed in {bootstrap_duration:.1f}s")
    log_gpu_memory()

    # Shared val stream for server-side evaluation
    val_stream = DetectionStream(
        manifest_path=manifest_path,
        split="val",
        transform=val_transform,
        min_box_area=config.min_box_area,
        frame_range=(0, config.val_frame_limit) if config.val_frame_limit else None,
        target_classes=config.target_classes,
        verbose=False,
    )

    # Remaining train stream (after bootstrap) is partitioned across clients
    train_stream_for_len = DetectionStream(
        manifest_path=manifest_path,
        split="train",
        transform=train_transform,
        augmentation=None,
        min_box_area=config.min_box_area,
        frame_range=(config.bootstrap_frames, None),
        target_classes=config.target_classes,
        verbose=False,
    )
    stream_frames = len(train_stream_for_len)

    if config.partition_strategy == "domain_aligned":
        import json
        with open(manifest_path) as _mf:
            _manifest_data = json.load(_mf)
        _ordering = _manifest_data.get("ordering", {})
        _block_order = _ordering.get("block_order", [])
        _block_sizes = _ordering.get("block_sizes", {})
        if not _block_order or not _block_sizes:
            raise ValueError(
                "domain_aligned partitioning requires a manifest with "
                "ordering.block_order and ordering.block_sizes"
            )
        if config.domain_client_groups is None:
            raise ValueError(
                "domain_client_groups must be set when "
                "partition_strategy='domain_aligned'"
            )
        if len(config.domain_client_groups) != config.num_clients:
            raise ValueError(
                f"domain_client_groups has {len(config.domain_client_groups)} "
                f"entries but num_clients={config.num_clients}"
            )
        partitions = partition_frames_by_domain(
            block_order=_block_order,
            block_sizes=_block_sizes,
            client_groups=config.domain_client_groups,
        )
        print(f"\nDomain-aligned partitioning ({len(_block_order)} blocks -> "
              f"{config.num_clients} clients)")
    else:
        partitions = partition_frames(
            num_frames=stream_frames,
            num_clients=config.num_clients,
            strategy=config.partition_strategy,
        )

    print("\nClient partitions (post-bootstrap stream indices):")
    for cid in range(config.num_clients):
        s, e = partitions[cid]
        print(f"  client_{cid}: [{s}, {e}) size={e - s}")

    # Frozen scoring model -- snapshot of bootstrap model for stable novelty scores.
    scoring_model: Optional[nn.Module] = None
    if config.filter_policy == "distribution":
        scoring_model = copy.deepcopy(global_model)
        scoring_model.eval()
        for p in scoring_model.parameters():
            p.requires_grad = False
        print("Created frozen scoring model for stable novelty measurement")

    # Per-client policy state (kept across rounds for threshold/history continuity).
    # bootstrap_scores are converted to a plain list for the filter constructor.
    _bs_list = bootstrap_scores.tolist() if bootstrap_scores is not None else None
    client_policies = [
        create_filter_policy(
            config,
            bootstrap_mean=embedding_mean,
            bootstrap_cov=embedding_cov,
            scoring_model=scoring_model,
            bootstrap_scores=_bs_list,
            reservoir_seed_override=config.seed + 1000 * (cid + 1),
        )
        for cid in range(config.num_clients)
    ]

    fed_logger = FederatedMetricsLogger(
        log_dir=run_dir,
        num_clients=config.num_clients,
        task="detection",
        class_names=class_mapping.names,
    )
    decisions_logger = FederatedDecisionsLogger(log_dir=run_dir)

    # Adaptive filter refresh setup (shared across clients).  One refresh
    # applies the same scoring model + reference + threshold to every
    # client's policy.  Disabled when scoring_refresh_every_rounds == 0.
    adaptive_refresh_enabled = (
        config.filter_policy == "distribution"
        and config.scoring_refresh_every_rounds > 0
    )
    scoring_refresher: Optional[ScoringRefresher] = None
    refresh_records: List[RefreshRecord] = []
    fleet_reference_size: int = 0
    pool_rng: Optional[random.Random] = None
    if adaptive_refresh_enabled:
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
        )
        fleet_reference_size = max(
            config.scoring_refresh_window_size,
            config.scoring_refresh_reservoir_size,
        )
        pool_rng = random.Random(config.seed + 7919)
        print(
            "Adaptive filter refresh enabled: every "
            f"{config.scoring_refresh_every_rounds} round(s), "
            f"window={config.scoring_refresh_window_size}, "
            f"reservoir={config.scoring_refresh_reservoir_size}, "
            f"fleet reference size={fleet_reference_size}"
        )

    elapsed_seconds = lambda: (datetime.now() - start_time).total_seconds()
    global_state = global_model.state_dict()
    final_metrics: Dict[str, float] = {}

    for round_idx in range(1, config.num_rounds + 1):
        print("\n" + "-" * 60)
        print(f"Round {round_idx}/{config.num_rounds}")
        print("-" * 60)

        local_state_dicts: List[Dict[str, torch.Tensor]] = []
        local_weights: List[float] = []
        client_results: List[Dict[str, int]] = []

        for cid in range(config.num_clients):
            start_i, end_i = partitions[cid]
            local_start = start_i + (round_idx - 1) * config.local_items_per_round
            local_end = min(local_start + config.local_items_per_round, end_i)

            if local_start >= local_end:
                client_results.append(
                    {
                        "items_processed": 0,
                        "items_accepted": 0,
                        "items_rejected": 0,
                        "optimizer_steps": 0,
                    }
                )
                print(f"  client_{cid}: no remaining data")
                continue

            client_model = build_detector_from_config(config).to(device)
            client_model.load_state_dict(global_state)

            client_optimizer = torch.optim.Adam(
                [p for p in client_model.parameters() if p.requires_grad],
                lr=config.streaming_lr,
                weight_decay=config.weight_decay,
            )

            # Local slice is offset by bootstrap_frames in full train stream.
            stream_slice = (
                config.bootstrap_frames + local_start,
                config.bootstrap_frames + local_end,
            )
            client_stream = DetectionStream(
                manifest_path=manifest_path,
                split="train",
                transform=train_transform,
                augmentation=None,
                min_box_area=config.min_box_area,
                frame_range=stream_slice,
                target_classes=config.target_classes,
                verbose=False,
            )

            client_buffer = TrainingBuffer(capacity=config.buffer_capacity)

            def _log_decision(
                global_idx: int,
                frame_id: str,
                action: str,
                filter_metric: str,
                filter_score: float,
                filter_threshold: Optional[float],
                categories,
                _round_idx: int = round_idx,
                _cid: int = cid,
            ) -> None:
                decisions_logger.log_decision(
                    round_idx=_round_idx,
                    client_id=_cid,
                    global_idx=global_idx,
                    frame_id=frame_id,
                    action=action,
                    filter_metric=filter_metric,
                    filter_score=filter_score,
                    filter_threshold=filter_threshold,
                    categories=categories,
                )

            result = train_on_stream(
                model=client_model,
                stream=client_stream,
                optimizer=client_optimizer,
                filter_policy=client_policies[cid],
                training_buffer=client_buffer,
                device=device,
                max_grad_norm=config.max_grad_norm,
                train_steps_per_buffer=config.train_steps_per_buffer,
                buffer_training_mode=config.buffer_training_mode,
                local_epochs_per_buffer=config.local_epochs_per_buffer,
                mini_batch_size=config.mini_batch_size,
                shuffle_buffer_each_epoch=config.shuffle_buffer_each_epoch,
                metrics_logger=None,
                eval_fn=None,
                progress_bar=False,
                total_items=len(client_stream),
                scaler=scaler,
                decision_callback=_log_decision,
            )

            accepted_items = int(result.items_accepted)
            rejected_items = int(result.items_rejected)
            opt_steps = int(result.optimizer_steps)
            if accepted_items > 0:
                local_state_dicts.append(client_model.state_dict())
                # Weight by amount of accepted (thus actually trained) data.
                local_weights.append(float(accepted_items))
            client_results.append(
                {
                    "items_processed": int(result.items_processed),
                    "items_accepted": accepted_items,
                    "items_rejected": rejected_items,
                    "optimizer_steps": opt_steps,
                }
            )
            print(
                f"  client_{cid}: processed={result.items_processed}, "
                f"accepted={result.items_accepted}, rejected={result.items_rejected}, "
                f"opt_steps={result.optimizer_steps}"
            )

        if local_state_dicts:
            global_state = fedavg(local_state_dicts, local_weights)
            global_model.load_state_dict(global_state)
        else:
            print("  WARNING: no client updates this round.")

        if (
            adaptive_refresh_enabled
            and scoring_refresher is not None
            and round_idx % config.scoring_refresh_every_rounds == 0
        ):
            dist_policies = [
                p for p in client_policies if isinstance(p, DistributionBasedPolicy)
            ]
            pooled_window = pool_recent_accepted(
                dist_policies, fleet_reference_size, rng=pool_rng,
            )
            record = scoring_refresher.refresh(
                live_model=global_model,
                policies=dist_policies,
                accepted_frame_ids=pooled_window,
                trigger="federated_round",
                trigger_count=round_idx,
            )
            refresh_records.append(record)
            print(
                f"  [refresh #{record.refresh_idx}] window={record.window_size}, "
                f"ref={record.reference_size}, "
                f"thr: {record.threshold_before:.3f} -> {record.threshold_after:.3f}, "
                f"{record.duration_seconds:.1f}s"
            )

        eval_metrics = None
        if round_idx % max(config.eval_every_n_rounds, 1) == 0:
            eval_metrics = evaluate_detection(
                global_model,
                val_stream,
                device,
                score_threshold=config.score_threshold,
                use_amp=config.use_amp,
            )
            final_metrics = eval_metrics
            print(
                f"  Server eval: mAP={eval_metrics.get('mAP', 0.0):.4f}, "
                f"mAP@50={eval_metrics.get('mAP_50', 0.0):.4f}"
            )

        fed_logger.log_round(
            round_idx=round_idx,
            eval_metrics=eval_metrics,
            client_results=client_results,
            elapsed=elapsed_seconds(),
        )

    # Final eval if last round wasn't evaluated
    if not final_metrics:
        final_metrics = evaluate_detection(
            global_model,
            val_stream,
            device,
            score_threshold=config.score_threshold,
            use_amp=config.use_amp,
        )

    print("\n" + "=" * 60)
    print("Federated training complete")
    print(f"  Final mAP:    {final_metrics.get('mAP', 0.0):.4f}")
    print(f"  Final mAP@50: {final_metrics.get('mAP_50', 0.0):.4f}")
    if adaptive_refresh_enabled:
        print(f"  Refreshes:    {len(refresh_records)}")
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

    torch.save(
        {
            "model_state_dict": global_model.state_dict(),
            "config": config.__dict__,
            "final_metrics": final_metrics,
            "partitions": partitions,
        },
        run_dir / "final_model.pt",
    )

    dataset_info = {
        "bootstrap_frames": config.bootstrap_frames,
        "stream_frames": stream_frames,
        "val_frames": len(val_stream),
        "num_clients": config.num_clients,
        "num_rounds": config.num_rounds,
        "partition_strategy": config.partition_strategy,
        "local_items_per_round": config.local_items_per_round,
        "bootstrap_reused": bootstrap_source is not None,
    }
    if bootstrap_source is not None:
        dataset_info["bootstrap_source"] = str(bootstrap_source)

    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        end_time=datetime.now(),
        best_metric=final_metrics.get("mAP", 0.0),
        metric_key="final_val_mAP",
        dataset_info=dataset_info,
        repo_path=PROJECT_ROOT,
    )

    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated streaming detection experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--bootstrap-run-dir",
        type=str,
        default=None,
        help="Reuse bootstrap model + embeddings from this run directory "
        "(overrides bootstrap_run_dir in config).",
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
    config = FederatedDetectionConfig.from_yaml(config_path)
    if args.bootstrap_run_dir:
        config.bootstrap_run_dir = args.bootstrap_run_dir
    if args.seed is not None:
        config.seed = args.seed

    if config.num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if config.num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")
    if config.local_items_per_round < 1:
        raise ValueError("local_items_per_round must be >= 1")

    main(config, config_path, command)
