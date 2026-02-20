"""
Federated streaming classification experiment.

Simulates federated learning with multiple clients, each training on its
own partition of ZOD sequences in temporal order. The server aggregates
client models via FedAvg after each round and evaluates on a shared
validation stream.

Parallel to streaming_classification.py but with federated rounds instead
of a single-pass stream.

Usage:
    python experiments/federated_streaming_classification.py --config configs/federated_streaming_classification.yaml
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import yaml

# Suppress NVML warning (cosmetic, GPU still works fine)
warnings.filterwarnings("ignore", message="Can't initialize NVML")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from stream_active_fl.core import StreamingDataset, get_classification_transforms
from stream_active_fl.core.partitioning import ClientStream, partition_sequences
from stream_active_fl.evaluation import evaluate_streaming_classification
from stream_active_fl.logging import FederatedMetricsLogger, create_run_dir, save_run_info
from stream_active_fl.memory import ReplayBuffer
from stream_active_fl.models import Classifier
from stream_active_fl.policies import FilterPolicy, create_filter_policy
from stream_active_fl.training import RunningPosWeight, train_on_classification_stream
from stream_active_fl.training.federated import fedavg
from stream_active_fl.utils import set_seed


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FederatedStreamingClassificationConfig:
    """Federated streaming classification experiment configuration."""

    # Paths
    dataset_root: str = "/mnt/pr_2018_scaleout_workdir/ZOD256/ZOD_640x360"
    annotations_dir: str = "data/annotations_640x360"
    output_dir: str = "outputs/federated_streaming_classification"

    # Dataset
    target_category: int = 0
    min_score: float = 0.5
    subsample_steps: int = 5

    # Model
    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = True
    dropout: float = 0.0
    load_checkpoint: Optional[str] = None

    # --- Federated settings ---
    num_clients: int = 3
    num_rounds: int = 10
    items_per_round: int = 1000
    partition_strategy: Literal["uniform", "contiguous"] = "uniform"

    # --- Local training (per-client, per-round) ---
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.0
    accumulation_steps: int = 1

    # Class imbalance
    pos_weight: str | float = "running"

    # Filtering policy (each client gets its own instance)
    filter_policy: Literal["none", "difficulty", "topk"] = "none"
    tau_teacher: float = 0.0
    store_gated: bool = False
    adaptive: bool = True
    train_fraction: float = 0.3
    loss_window_size: int = 500
    warmup_items: int = 200
    tau_loss: float = 0.5
    store_skipped: bool = False
    topk_window_size: int = 100
    topk_k: int = 30

    # Replay buffer (per-client, persists across rounds)
    use_replay: bool = True
    replay_capacity: int = 2000
    replay_batch_size: int = 32
    replay_admission_policy: Literal["fifo", "random", "reservoir"] = "reservoir"
    replay_weight: float = 0.5

    # Evaluation
    eval_every_n_rounds: int = 1

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FederatedStreamingClassificationConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Per-client persistent state
# =============================================================================


@dataclass
class ClientState:
    """Persistent state for a single simulated client.

    Bundles the client's data stream, filter policy, replay buffer, and
    class-imbalance estimator so they survive across federated rounds.
    """

    client_id: int
    stream: ClientStream
    filter_policy: FilterPolicy
    replay_buffer: Optional[ReplayBuffer]
    running_pos_weight: Optional[RunningPosWeight]
    pos_weight_tensor: Optional[torch.Tensor]

    total_items_processed: int = 0
    total_items_trained: int = 0


# =============================================================================
# Main
# =============================================================================


def main(
    config: FederatedStreamingClassificationConfig,
    config_path: Path,
    command: str,
) -> None:
    """Run federated streaming classification experiment."""

    start_time = datetime.now()

    print("=" * 60)
    print("Federated Streaming Classification")
    print("=" * 60)

    # Setup
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve paths and create run directory
    annotations_dir = PROJECT_ROOT / config.annotations_dir
    base_output_dir = PROJECT_ROOT / config.output_dir
    run_dir = create_run_dir(base_output_dir)
    print(f"Run directory: {run_dir}")

    shutil.copy(config_path, run_dir / "config.yaml")

    # Transforms
    train_transform, val_transform = get_classification_transforms()

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    print("\nLoading streaming datasets...")
    train_stream = StreamingDataset(
        dataset_root=config.dataset_root,
        annotations_dir=annotations_dir,
        split="train",
        transform=train_transform,
        target_category=config.target_category,
        min_score=config.min_score,
        subsample_steps=config.subsample_steps,
        verbose=True,
    )

    val_stream = StreamingDataset(
        dataset_root=config.dataset_root,
        annotations_dir=annotations_dir,
        split="val",
        transform=val_transform,
        target_category=config.target_category,
        min_score=config.min_score,
        subsample_steps=config.subsample_steps,
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # Partition sequences across clients
    # -------------------------------------------------------------------------
    num_sequences = len(train_stream.sequence_metadata)
    partitions = partition_sequences(
        num_sequences=num_sequences,
        num_clients=config.num_clients,
        strategy=config.partition_strategy,
        seed=config.seed,
    )

    print(f"\nPartitioned {num_sequences} sequences across {config.num_clients} clients:")
    for i, part in enumerate(partitions):
        n_items = sum(
            train_stream.sequence_metadata[s]["num_subsampled_frames"] for s in part
        )
        print(f"  Client {i}: {len(part)} sequences, {n_items} items")

    # Log partition heterogeneity (useful for non-IID analysis)
    print("\nPartition heterogeneity (per-client positive rates):")
    partition_stats = []
    for i, part in enumerate(partitions):
        cs_temp = ClientStream(train_stream, part)
        seq_stats = cs_temp.get_sequence_stats()
        total_frames = sum(s["num_frames"] for s in seq_stats)
        total_positive = sum(s["num_positive"] for s in seq_stats)
        pos_rate = total_positive / max(total_frames, 1)
        partition_stats.append(pos_rate)
        print(f"  Client {i}: positive_rate={pos_rate:.4f} "
              f"({total_positive}/{total_frames})")
    spread = max(partition_stats) - min(partition_stats) if partition_stats else 0.0
    print(f"  Spread (max - min): {spread:.4f}")

    # -------------------------------------------------------------------------
    # Global model
    # -------------------------------------------------------------------------
    print("\nInitializing global model...")
    global_model = Classifier(
        backbone=config.backbone,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        dropout=config.dropout,
    )

    if config.load_checkpoint:
        checkpoint_path = PROJECT_ROOT / config.load_checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        global_model.load_state_dict(ckpt["model_state_dict"])

    global_model = global_model.to(device)
    print(global_model)

    # -------------------------------------------------------------------------
    # Per-client persistent state
    # -------------------------------------------------------------------------
    clients: list[ClientState] = []
    for client_id, seq_indices in enumerate(partitions):
        client_stream = ClientStream(train_stream, seq_indices)
        filter_policy = create_filter_policy(config)

        replay_buffer = None
        if config.use_replay:
            replay_buffer = ReplayBuffer(
                capacity=config.replay_capacity,
                admission_policy=config.replay_admission_policy,
                device="cpu",
            )

        running_pw = None
        pw_tensor = None
        if config.pos_weight == "running":
            pw_tensor = torch.tensor([1.0], device=device)
            running_pw = RunningPosWeight(pw_tensor)

        clients.append(ClientState(
            client_id=client_id,
            stream=client_stream,
            filter_policy=filter_policy,
            replay_buffer=replay_buffer,
            running_pos_weight=running_pw,
            pos_weight_tensor=pw_tensor,
        ))

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    fl_logger = FederatedMetricsLogger(run_dir, config.num_clients, task="classification")

    dataset_info = {
        "train_total": len(train_stream),
        "val_total": len(val_stream),
        "num_clients": config.num_clients,
        "num_rounds": config.num_rounds,
        "items_per_round": config.items_per_round,
        "partition_sizes": [len(p) for p in partitions],
        "partition_positive_rates": partition_stats,
    }
    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        dataset_info=dataset_info,
        repo_path=PROJECT_ROOT,
    )

    # -------------------------------------------------------------------------
    # Initial evaluation (before any training)
    # -------------------------------------------------------------------------
    print("\nInitial evaluation (round 0)...")
    init_metrics = evaluate_streaming_classification(
        global_model, val_stream, nn.BCEWithLogitsLoss(), device,
    )
    print(f"  Val F1: {init_metrics['f1']:.4f}  (before training)")
    fl_logger.log_round(
        round_idx=0,
        eval_metrics=init_metrics,
        client_results=[{"items_processed": 0, "items_trained": 0}] * config.num_clients,
        elapsed=0.0,
    )

    best_f1 = init_metrics["f1"]
    best_round = 0

    # Pre-allocate a single client model to avoid deepcopy every round.
    # Since clients train sequentially, we reuse this model by loading the
    # global state dict at the start of each client's turn.
    client_model = copy.deepcopy(global_model)

    # -------------------------------------------------------------------------
    # Federated rounds
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Starting federated training: {config.num_rounds} rounds, "
          f"{config.num_clients} clients, {config.items_per_round} items/round/client")
    print("=" * 60)

    wall_start = time.time()

    for round_idx in range(1, config.num_rounds + 1):
        round_start = time.time()

        print(f"\n--- Round {round_idx}/{config.num_rounds} ---")

        client_state_dicts = []
        client_weights = []
        round_client_results = []

        global_state = global_model.state_dict()

        for cs in clients:
            if cs.stream.exhausted:
                print(f"  Client {cs.client_id}: stream exhausted, skipping")
                round_client_results.append({"items_processed": 0, "items_trained": 0})
                continue

            # Reset client model to global state (much cheaper than deepcopy)
            client_model.load_state_dict(global_state)

            # Fresh optimizer each round (standard FedAvg)
            client_optimizer = torch.optim.Adam(
                client_model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            # Build criterion with this client's pos_weight tensor
            if cs.pos_weight_tensor is not None:
                criterion = nn.BCEWithLogitsLoss(pos_weight=cs.pos_weight_tensor)
            else:
                pw_val = float(config.pos_weight)
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pw_val], device=device)
                )

            # Train on this client's stream
            result = train_on_classification_stream(
                model=client_model,
                stream=cs.stream,
                criterion=criterion,
                optimizer=client_optimizer,
                filter_policy=cs.filter_policy,
                device=device,
                replay_buffer=cs.replay_buffer,
                replay_batch_size=config.replay_batch_size,
                replay_weight=config.replay_weight,
                max_grad_norm=config.max_grad_norm,
                accumulation_steps=config.accumulation_steps,
                max_items=config.items_per_round,
                running_pos_weight=cs.running_pos_weight,
                filter_computes_forward=(config.filter_policy != "none"),
                progress_bar=False,
            )

            cs.total_items_processed += result.items_processed
            cs.total_items_trained += result.items_trained

            print(
                f"  Client {cs.client_id}: "
                f"processed={result.items_processed}, "
                f"trained={result.items_trained}, "
                f"remaining~{cs.stream.remaining_items}"
            )

            round_client_results.append({
                "items_processed": result.items_processed,
                "items_trained": result.items_trained,
            })

            if result.items_trained > 0:
                # Clone tensors: state_dict() returns views into model params,
                # which get overwritten when we load_state_dict for the next client.
                client_state_dicts.append(
                    {k: v.clone() for k, v in client_model.state_dict().items()}
                )
                client_weights.append(float(result.items_trained))

        # Aggregate
        n_active = len(client_state_dicts)
        if client_state_dicts:
            aggregated = fedavg(client_state_dicts, weights=client_weights)
            global_model.load_state_dict(aggregated)
            print(f"  Aggregated {n_active}/{config.num_clients} client(s) "
                  f"(weights: {[f'{w:.0f}' for w in client_weights]})")
        else:
            print("  No clients trained this round, global model unchanged")

        # Evaluate
        eval_metrics = None
        if round_idx % config.eval_every_n_rounds == 0:
            eval_metrics = evaluate_streaming_classification(
                global_model, val_stream, nn.BCEWithLogitsLoss(), device,
            )
            f1 = eval_metrics["f1"]
            print(f"  Val F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_round = round_idx
                torch.save(
                    {
                        "model_state_dict": global_model.state_dict(),
                        "round": round_idx,
                        "f1": f1,
                    },
                    run_dir / "best_model.pt",
                )
                print(f"  Saved new best model (F1={f1:.4f})")

        elapsed = time.time() - wall_start
        fl_logger.log_round(round_idx, eval_metrics, round_client_results, elapsed)

        round_elapsed = time.time() - round_start
        print(f"  Round time: {round_elapsed:.1f}s")

        # Early stop if all clients exhausted
        if all(cs.stream.exhausted for cs in clients):
            print("\nAll client streams exhausted, stopping early.")
            break

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - wall_start

    print("\n" + "=" * 60)
    print("Federated training complete!")
    print(f"  Best val F1: {best_f1:.4f} (round {best_round})")
    print(f"  Total time:  {total_elapsed:.1f}s")
    print("-" * 60)
    for cs in clients:
        print(
            f"  Client {cs.client_id}: "
            f"total_processed={cs.total_items_processed}, "
            f"total_trained={cs.total_items_trained}, "
            f"stream_exhausted={cs.stream.exhausted}"
        )
    print("=" * 60)

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_streaming_classification(
        global_model, val_stream, nn.BCEWithLogitsLoss(), device,
    )
    print(f"  Val Loss: {final_metrics['loss']:.4f}")
    print(f"  Val Acc:  {final_metrics['accuracy']:.4f}")
    print(f"  Val F1:   {final_metrics['f1']:.4f}")

    # Save final model
    torch.save(
        {
            "model_state_dict": global_model.state_dict(),
            "config": config.__dict__,
            "final_metrics": final_metrics,
            "best_f1": best_f1,
            "best_round": best_round,
        },
        run_dir / "final_model.pt",
    )

    # Save final run info
    end_time = datetime.now()
    save_run_info(
        run_dir=run_dir,
        config=config,
        command=command,
        start_time=start_time,
        end_time=end_time,
        best_metric=best_f1,
        best_metric_name="val_f1",
        dataset_info=dataset_info,
        extra_info={"best_round": best_round, "final_metrics": final_metrics},
        repo_path=PROJECT_ROOT,
    )

    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated streaming classification training"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    command = " ".join(sys.argv)
    config = FederatedStreamingClassificationConfig.from_yaml(config_path)
    main(config, config_path, command)
