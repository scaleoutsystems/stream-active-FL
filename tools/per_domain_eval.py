"""Retrospective per-domain evaluation for streaming detection runs.

Walks a runs root (default: outputs/streaming/) and, for each run dir
containing best_model.pt and/or final_model.pt, runs inference on the
configured validation split, partitions the val frames by domain metadata
(time_of_day, road_condition, road_type), and writes a long-format
per_domain_eval.csv into the run directory.

One row per (checkpoint, dimension, bucket) tuple -- aggregate is recorded
as dimension='aggregate', bucket='all'.  Re-running the script skips runs
that already have an up-to-date per_domain_eval.csv unless --force is set.

Usage:
    python tools/per_domain_eval.py \
        --runs-root outputs/streaming/ \
        --manifest data/ZOD_frames_preprocessed/Frames_1600x480/manifest_cityday_curated_boot2000.json \
        --checkpoints best final

Optional filters:
    --variant-pattern 'cityday_curated'   glob on variant dir name
    --seed-pattern    'seed_42'           glob on seed dir name
    --skip-existing                        skip runs with existing CSV
    --force                                recompute even if CSV present
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stream_active_fl.core import (  # noqa: E402
    DetectionStream,
    build_class_mapping,
    get_detection_transforms,
    load_manifest,
)
from stream_active_fl.evaluation import (  # noqa: E402
    EXTENDED_DOMAIN_DIMS,
    attach_stream_blocks,
    evaluate_detection,
)
from stream_active_fl.runtime import (  # noqa: E402
    build_detector_from_config,
    resolve_manifest_path,
)


CHECKPOINT_FILENAMES: Dict[str, str] = {
    "best": "best_model.pt",
    "final": "final_model.pt",
}


@dataclass
class RunTarget:
    run_dir: Path
    variant: str
    seed: str
    timestamp: str


@dataclass
class _ModelConfigStub:
    """Minimal config object for build_detector_from_config."""

    num_classes: int
    trainable_backbone_layers: int
    image_min_size: int
    image_max_size: int
    pretrained_backbone: bool
    pretrained_detector: bool


def _discover_run_dirs(
    runs_root: Path,
    variant_pattern: Optional[str],
    seed_pattern: Optional[str],
    variants: Optional[List[str]] = None,
) -> List[RunTarget]:
    """Find <variant>/<seed>/<timestamp> run directories under runs_root.

    Args:
        runs_root: Root directory holding variant subdirs.
        variant_pattern: Substring that variant dir names must contain.
        seed_pattern: Substring that seed dir names must contain.
        variants: Exact variant dir names; takes precedence over variant_pattern.

    Returns:
        List of discovered RunTarget entries, in sorted order.
    """
    targets: List[RunTarget] = []
    if not runs_root.exists():
        return targets
    exact = set(variants) if variants else None
    for variant_dir in sorted(runs_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        if exact is not None:
            if variant_dir.name not in exact:
                continue
        elif variant_pattern and not fnmatch.fnmatch(variant_dir.name, f"*{variant_pattern}*"):
            continue
        for seed_dir in sorted(variant_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            if seed_pattern and not fnmatch.fnmatch(seed_dir.name, f"*{seed_pattern}*"):
                continue
            timestamps = sorted([d for d in seed_dir.iterdir() if d.is_dir()])
            for ts_dir in timestamps:
                targets.append(
                    RunTarget(
                        run_dir=ts_dir,
                        variant=variant_dir.name,
                        seed=seed_dir.name,
                        timestamp=ts_dir.name,
                    )
                )
    return targets


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Load config.yaml from a run dir as a plain dict."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {run_dir}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dict in {cfg_path}, got {type(cfg).__name__}")
    return cfg


def _build_model_from_run_config(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Instantiate a Detector sized to match the saved checkpoint.

    Mirrors the runtime override in experiments/streaming.py: when
    target_classes is set, the model's output dimensionality is driven by
    build_class_mapping(target_classes).num_classes (background + targets),
    not by the num_classes field written into config.yaml.
    """
    target_classes = cfg.get("target_classes")
    num_classes = int(cfg.get("num_classes", 11))
    if target_classes is not None:
        num_classes = build_class_mapping(target_classes).num_classes
    stub = _ModelConfigStub(
        num_classes=num_classes,
        trainable_backbone_layers=int(cfg.get("trainable_backbone_layers", 3)),
        image_min_size=int(cfg.get("image_min_size", 480)),
        image_max_size=int(cfg.get("image_max_size", 1600)),
        pretrained_backbone=bool(cfg.get("pretrained_backbone", True)),
        pretrained_detector=bool(cfg.get("pretrained_detector", True)),
    )
    return build_detector_from_config(stub)


def _load_state_dict(checkpoint_path: Path, model: torch.nn.Module) -> None:
    """Load a model state_dict from a training checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)


def _extract_domain_labels(
    stream: DetectionStream, dims: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Return frame_id -> domain metadata, for robust per-frame alignment.

    Only raw metadata keys are copied here; stream_block (a joint label
    derived from the manifest ordering strategy) is attached later via
    attach_stream_blocks.
    """
    labels: Dict[str, Dict[str, Any]] = {}
    for frame in stream.frames:
        fid = str(frame["frame_id"])
        labels[fid] = {dim: frame.get(dim) for dim in dims if dim != "stream_block"}
    return labels


def _load_manifest_strategy(manifest_path: Path) -> Optional[str]:
    """Read manifest's ordering strategy name, or None if missing/unreadable."""
    try:
        import json
        with open(manifest_path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    ordering = data.get("ordering") if isinstance(data, dict) else None
    if isinstance(ordering, dict):
        strat = ordering.get("strategy")
        return str(strat) if strat is not None else None
    return None


def _write_long_csv(
    csv_path: Path,
    rows: List[Dict[str, Any]],
    class_names: List[str],
) -> None:
    """Write long-format per-domain eval results, one row per bucket."""
    fieldnames = [
        "run_variant", "run_seed", "run_timestamp",
        "checkpoint", "checkpoint_mtime",
        "dimension", "bucket", "n_frames",
        "mAP", "mAP_50", "mAP_75",
        "total_predictions", "total_ground_truth",
    ] + [f"AP_{c}" for c in class_names]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _results_to_rows(
    metrics: Dict[str, float],
    run_meta: Dict[str, str],
    checkpoint: str,
    checkpoint_mtime: float,
    class_names: List[str],
    domain_dims: List[str],
) -> List[Dict[str, Any]]:
    """Turn a flat metrics dict into one row per (dimension, bucket)."""
    rows: List[Dict[str, Any]] = []

    agg_row: Dict[str, Any] = {
        **run_meta,
        "checkpoint": checkpoint,
        "checkpoint_mtime": f"{checkpoint_mtime:.0f}",
        "dimension": "aggregate",
        "bucket": "all",
        "n_frames": int(metrics.get("num_items", 0)),
        "mAP": metrics.get("mAP", float("nan")),
        "mAP_50": metrics.get("mAP_50", float("nan")),
        "mAP_75": metrics.get("mAP_75", float("nan")),
        "total_predictions": metrics.get("total_predictions", float("nan")),
        "total_ground_truth": metrics.get("total_ground_truth", float("nan")),
    }
    for cls in class_names:
        agg_row[f"AP_{cls}"] = metrics.get(f"AP_{cls}", float("nan"))
    rows.append(agg_row)

    for dim in domain_dims:
        prefix_n = f"n_{dim}_"
        buckets = sorted({
            k[len(prefix_n):]
            for k in metrics.keys()
            if k.startswith(prefix_n)
        })
        for bucket in buckets:
            tag = f"{dim}_{bucket}"
            row: Dict[str, Any] = {
                **run_meta,
                "checkpoint": checkpoint,
                "checkpoint_mtime": f"{checkpoint_mtime:.0f}",
                "dimension": dim,
                "bucket": bucket,
                "n_frames": int(metrics.get(f"n_{tag}", 0)),
                "mAP": metrics.get(f"mAP_{tag}", float("nan")),
                "mAP_50": metrics.get(f"mAP_50_{tag}", float("nan")),
                "mAP_75": metrics.get(f"mAP_75_{tag}", float("nan")),
                "total_predictions": metrics.get(f"total_predictions_{tag}", float("nan")),
                "total_ground_truth": metrics.get(f"total_ground_truth_{tag}", float("nan")),
            }
            for cls in class_names:
                row[f"AP_{cls}"] = metrics.get(f"AP_{cls}_{tag}", float("nan"))
            rows.append(row)
    return rows


def _csv_is_fresh(csv_path: Path, checkpoint_paths: List[Path]) -> bool:
    """True iff csv_path exists and is newer than every provided checkpoint."""
    if not csv_path.exists():
        return False
    csv_mtime = csv_path.stat().st_mtime
    for ckpt in checkpoint_paths:
        if ckpt.exists() and ckpt.stat().st_mtime > csv_mtime:
            return False
    return True


def evaluate_run(
    target: RunTarget,
    manifest_path: Path,
    checkpoints: List[str],
    device: torch.device,
    use_amp: bool,
    domain_dims: List[str],
    score_threshold: Optional[float] = None,
    min_box_area: Optional[float] = None,
    skip_existing: bool = False,
    force: bool = False,
) -> Optional[Path]:
    """Run per-domain eval on one run directory; returns the CSV path."""
    cfg = _load_run_config(target.run_dir)

    ckpt_paths: List[Path] = []
    for name in checkpoints:
        fname = CHECKPOINT_FILENAMES[name]
        p = target.run_dir / fname
        if p.exists():
            ckpt_paths.append(p)

    if not ckpt_paths:
        print(f"[skip] {target.run_dir}: no checkpoints present")
        return None

    csv_path = target.run_dir / "per_domain_eval.csv"
    if not force and skip_existing and _csv_is_fresh(csv_path, ckpt_paths):
        print(f"[skip] {target.run_dir}: per_domain_eval.csv is up to date")
        return csv_path

    _, val_transform = get_detection_transforms()
    target_classes = cfg.get("target_classes")
    val_stream = DetectionStream(
        manifest_path=manifest_path,
        split="val",
        transform=val_transform,
        min_box_area=float(min_box_area if min_box_area is not None else cfg.get("min_box_area", 64.0)),
        target_classes=target_classes,
        verbose=False,
    )
    # Raw marginal metadata; stream_block is derived from the manifest.
    raw_dims = [d for d in domain_dims if d != "stream_block"]
    domain_labels = _extract_domain_labels(val_stream, raw_dims)
    effective_dims = list(raw_dims)
    if "stream_block" in domain_dims:
        strategy = _load_manifest_strategy(manifest_path)
        if attach_stream_blocks(domain_labels, list(val_stream.frames), strategy):
            effective_dims.append("stream_block")
    class_names = list(val_stream.class_mapping.names)
    score_th = float(
        score_threshold if score_threshold is not None else cfg.get("score_threshold", 0.3)
    )

    rows: List[Dict[str, Any]] = []
    run_meta = {
        "run_variant": target.variant,
        "run_seed": target.seed,
        "run_timestamp": target.timestamp,
    }

    for ckpt_path in ckpt_paths:
        checkpoint_name = {v: k for k, v in CHECKPOINT_FILENAMES.items()}[ckpt_path.name]
        t0 = time.time()
        print(
            f"[eval] {target.variant}/{target.seed}/{target.timestamp} :: {checkpoint_name}"
        )
        model = _build_model_from_run_config(cfg)
        _load_state_dict(ckpt_path, model)
        model.to(device)
        model.eval()

        metrics = evaluate_detection(
            model=model,
            val_stream=val_stream,
            device=device,
            score_threshold=score_th,
            use_amp=use_amp,
            domain_labels=domain_labels,
            domain_dims=effective_dims,
        )

        rows.extend(
            _results_to_rows(
                metrics=metrics,
                run_meta=run_meta,
                checkpoint=checkpoint_name,
                checkpoint_mtime=ckpt_path.stat().st_mtime,
                class_names=class_names,
                domain_dims=effective_dims,
            )
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"       mAP={metrics['mAP']:.4f}"
            f"  night={metrics.get('mAP_time_of_day_night', float('nan')):.4f}"
            f"  snow={metrics.get('mAP_road_condition_snow', float('nan')):.4f}"
            f"  ({time.time() - t0:.0f}s)"
        )

    _write_long_csv(csv_path, rows, class_names)
    print(f"       wrote {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrospective per-domain evaluation for streaming runs",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/streaming"),
        help="Root directory holding <variant>/<seed>/<timestamp> runs",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Manifest JSON path (resolved relative to project root if needed)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["best", "final"],
        choices=sorted(CHECKPOINT_FILENAMES.keys()),
        help="Which checkpoint(s) to evaluate per run",
    )
    parser.add_argument(
        "--variant-pattern", type=str, default=None,
        help="Only evaluate variants whose dir name contains this substring",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Exact variant dir names (takes precedence over --variant-pattern)",
    )
    parser.add_argument(
        "--seed-pattern", type=str, default=None,
        help="Only evaluate seed dirs whose name contains this substring",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip runs whose per_domain_eval.csv is newer than the checkpoints",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even when per_domain_eval.csv is already up to date",
    )
    parser.add_argument(
        "--domain-dims", nargs="+", default=list(EXTENDED_DOMAIN_DIMS),
        help="Metadata keys to bucketize on.  stream_block is derived "
             "from the manifest ordering strategy (cityday_curated_blocks)",
    )
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--min-box-area", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    runs_root = args.runs_root if args.runs_root.is_absolute() else PROJECT_ROOT / args.runs_root
    manifest_path = resolve_manifest_path(PROJECT_ROOT, args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    targets = _discover_run_dirs(
        runs_root, args.variant_pattern, args.seed_pattern, variants=args.variants,
    )
    if not targets:
        print(f"No run directories discovered under {runs_root}")
        return

    print(f"Runs root     : {runs_root}")
    print(f"Manifest      : {manifest_path}")
    print(f"Device        : {device}  (amp={use_amp})")
    print(f"Checkpoints   : {args.checkpoints}")
    print(f"Domain dims   : {args.domain_dims}")
    print(f"Discovered    : {len(targets)} run dirs")
    print()

    ok, fail = 0, 0
    for i, target in enumerate(targets, 1):
        print(f"--- [{i}/{len(targets)}] {target.variant}/{target.seed}/{target.timestamp} ---")
        try:
            evaluate_run(
                target=target,
                manifest_path=manifest_path,
                checkpoints=args.checkpoints,
                device=device,
                use_amp=use_amp,
                domain_dims=list(args.domain_dims),
                score_threshold=args.score_threshold,
                min_box_area=args.min_box_area,
                skip_existing=args.skip_existing,
                force=args.force,
            )
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"  ERROR: {exc!r}")
    print()
    print(f"Done.  ok={ok}  fail={fail}  total={len(targets)}")


if __name__ == "__main__":
    main()
