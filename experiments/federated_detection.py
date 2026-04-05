"""
Simulated federated learning for streaming object detection.

Pipeline:
  Phase 1 (Bootstrap): optional shared bootstrap model on first N frames.
  Phase 2 (Federated): partition the remaining stream across clients, run
      local streaming updates, aggregate with FedAvg each round.

Usage:
    python experiments/federated_detection.py --config configs/federated_no_filter.yaml
"""

from __future__ import annotations

import argparse
import csv
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
    partition_frames,
)
from stream_active_fl.evaluation import evaluate_detection
from stream_active_fl.experiment import (
    build_detector_from_config,
    load_dataclass_config,
    resolve_manifest_path,
    setup_run_dir,
)
from stream_active_fl.logging import (
    FederatedMetricsLogger,
    log_gpu_memory,
    save_run_info,
)
from stream_active_fl.memory import TrainingBuffer
from stream_active_fl.policies import create_filter_policy
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
    # "uniform" currently aliases "contiguous" (kept for backward compatibility).
    partition_strategy: Literal["contiguous", "uniform"] = "contiguous"

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

    # Augmentation
    augment: bool = True
    hflip_prob: float = 0.5
    color_jitter: bool = True

    # Filtering policy (per-client)
    # Reuses the same streaming filter policies as centralized experiments.
    filter_policy: Literal["none", "random", "distribution", "uncertainty", "gradient_norm"] = "none"
    accept_fraction: float = 0.3
    warmup_items: int = 200
    score_window_size: int = 500
    distribution_mode: Literal["mahalanobis", "cosine", "knn"] = "mahalanobis"
    embedding_buffer_size: int = 1000
    knn_k: int = 10
    update_distribution_stats: bool = True
    confidence_threshold: float = 0.1
    top_k_detections: int = 5
    norm_window_size: int = 500

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
) -> tuple[nn.Module, Optional[torch.Tensor], Optional[torch.Tensor], int, Optional[Path]]:
    requires_bootstrap_embeddings = (config.filter_policy == "distribution")
    bootstrap_source: Optional[Path] = None
    embedding_mean: Optional[torch.Tensor] = None
    embedding_cov: Optional[torch.Tensor] = None
    embedding_count = 0

    if config.bootstrap_run_dir:
        p = Path(config.bootstrap_run_dir)
        bootstrap_source = p if p.is_absolute() else PROJECT_ROOT / p

    if bootstrap_source is not None:
        model_path = bootstrap_source / "bootstrap_model.pt"
        embed_path = bootstrap_source / "bootstrap_embeddings.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Bootstrap model not found: {model_path}")
        if requires_bootstrap_embeddings and not embed_path.exists():
            raise FileNotFoundError(
                f"Bootstrap embeddings required for distribution policy, not found: {embed_path}"
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

        if requires_bootstrap_embeddings:
            embed_data = torch.load(embed_path, map_location="cpu", weights_only=True)
            if "count" not in embed_data:
                raise KeyError(
                    "bootstrap_embeddings.pt is missing required key 'count'. "
                    "Regenerate bootstrap embeddings with current code."
                )
            embedding_mean = embed_data["mean"]
            embedding_cov = embed_data["cov"]
            embedding_count = int(embed_data["count"])
            assert embedding_mean is not None and embedding_cov is not None
            print(
                "Loaded embeddings:"
                f" mean {embedding_mean.shape}, cov {embedding_cov.shape}, n={embedding_count}"
            )

        shutil.copy(model_path, run_dir / "bootstrap_model.pt")
        if requires_bootstrap_embeddings:
            shutil.copy(embed_path, run_dir / "bootstrap_embeddings.pt")
        (run_dir / "bootstrap_source.txt").write_text(str(bootstrap_source))
        return model, embedding_mean, embedding_cov, embedding_count, bootstrap_source

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
        print("\nCollecting embedding statistics...")
        embed_loader = torch.utils.data.DataLoader(
            bootstrap_dataset,
            batch_size=config.bootstrap_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=detection_collate,
            worker_init_fn=worker_init_fn,
            pin_memory=device.type == "cuda",
        )
        embedding_mean, embedding_cov, embedding_count = collect_embeddings(model, embed_loader, device)
        torch.save(
            {"mean": embedding_mean, "cov": embedding_cov, "count": embedding_count},
            run_dir / "bootstrap_embeddings.pt",
        )
        print(
            f"Embedding stats collected: mean {embedding_mean.shape}, "
            f"cov {embedding_cov.shape}, n={embedding_count}"
        )

    return model, embedding_mean, embedding_cov, embedding_count, None


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
    run_dir = setup_run_dir(PROJECT_ROOT, config.output_dir, config_path)
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
    global_model, embedding_mean, embedding_cov, embedding_count, bootstrap_source = _bootstrap_or_reuse(
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
        augmentation=train_augmentation,
        min_box_area=config.min_box_area,
        frame_range=(config.bootstrap_frames, None),
        target_classes=config.target_classes,
        verbose=False,
    )
    stream_frames = len(train_stream_for_len)
    partitions = partition_frames(
        num_frames=stream_frames,
        num_clients=config.num_clients,
        strategy=config.partition_strategy,
    )

    print("\nClient partitions (post-bootstrap stream indices):")
    for cid in range(config.num_clients):
        s, e = partitions[cid]
        print(f"  client_{cid}: [{s}, {e}) size={e - s}")

    # Per-client policy state (kept across rounds for threshold/history continuity)
    client_policies = [
        create_filter_policy(
            config,
            bootstrap_mean=embedding_mean,
            bootstrap_cov=embedding_cov,
            bootstrap_count=embedding_count,
        )
        for _ in range(config.num_clients)
    ]

    fed_logger = FederatedMetricsLogger(
        log_dir=run_dir,
        num_clients=config.num_clients,
        task="detection",
        class_names=class_mapping.names,
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
                augmentation=train_augmentation,
                min_box_area=config.min_box_area,
                frame_range=stream_slice,
                target_classes=config.target_classes,
                verbose=False,
            )

            client_buffer = TrainingBuffer(capacity=config.buffer_capacity)
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
                novelty_tracker=None,
                progress_bar=False,
                total_items=len(client_stream),
                scaler=scaler,
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
    print("=" * 60)

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
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    command = " ".join(sys.argv)
    config = FederatedDetectionConfig.from_yaml(config_path)
    if args.bootstrap_run_dir:
        config.bootstrap_run_dir = args.bootstrap_run_dir

    if config.num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if config.num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")
    if config.local_items_per_round < 1:
        raise ValueError("local_items_per_round must be >= 1")

    main(config, config_path, command)
