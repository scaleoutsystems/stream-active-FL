#!/usr/bin/env python3
"""
Generate manifest files for streaming experiments.

The base manifest is built by scanning a preprocessed data directory
(images/ and annotations/) and looking up ZOD metadata for timestamps
and train/val splits.  Ordering variants reorder the train frames for
different streaming strategies.

Outputs (written to <data-dir>/):
  manifest.json             Base manifest
  manifest_temporal.json    Train frames sorted by timestamp
  manifest_road_type.json   Train frames grouped by road type
  manifest_urban_rural.json Train frames grouped into urban/rural/highway

Usage:
  Generate everything:
    python tools/preprocessing/build_manifests.py \\
        --data-dir data/Frames_1600x480 \\
        --zod-root /path/to/zod

  Generate only the base manifest:
    python tools/preprocessing/build_manifests.py \\
        --data-dir data/Frames_1600x480 \\
        --zod-root /path/to/zod \\
        --variants base

  Generate only ordering variants (base manifest must already exist):
    python tools/preprocessing/build_manifests.py \\
        --data-dir data/Frames_1600x480 \\
        --variants temporal
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, cast

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CATEGORY_NAME_TO_ID, CROP_PARAMS, RESIZE_HEIGHT, RESIZE_WIDTH

ZodVersion = Literal["full", "mini"]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# ZOD metadata
# ---------------------------------------------------------------------------


def _load_zod_metadata(
    frame_ids: List[str],
    zod_root: str,
    zod_version: ZodVersion,
) -> Dict[str, Dict[str, Any]]:
    """Load timestamps, train/val splits, and road type from ZOD."""
    from zod import ZodFrames
    from zod.constants import TRAIN, VAL

    zod = ZodFrames(zod_root, zod_version)
    train_ids = zod.get_split(TRAIN)
    val_ids = zod.get_split(VAL)

    meta: Dict[str, Dict[str, Any]] = {}
    for fid in tqdm(frame_ids, desc="Loading ZOD metadata"):
        m = zod[fid].metadata
        timestamp = None
        if hasattr(m, "time") and m.time is not None:
            ts = m.time
            timestamp = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

        meta[fid] = {
            "timestamp": timestamp,
            "split": "train" if fid in train_ids else ("val" if fid in val_ids else "unknown"),
            "road_type": str(getattr(m, "road_type", "unknown") or "unknown"),
        }
    return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _temporal_key(frame: Dict[str, Any]) -> tuple[int, str, str]:
    """Sort key: frames with timestamps first, then by timestamp, then frame_id."""
    ts = frame.get("timestamp")
    if ts is None:
        return (1, "", frame.get("frame_id", ""))
    return (0, str(ts), frame.get("frame_id", ""))


def _replace_train_order(
    all_frames: List[Dict[str, Any]],
    reordered_train: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Substitute train frames with reordered_train, keeping val frames in place."""
    idx = 0
    out: List[Dict[str, Any]] = []
    for f in all_frames:
        if f.get("split") == "train":
            out.append(reordered_train[idx])
            idx += 1
        else:
            out.append(f)
    assert idx == len(reordered_train), "Train frame count mismatch."
    return out


def _urban_rural_bucket(road_type: str) -> str:
    """Map road_type to coarse urban/rural/highway/other bucket."""
    road = road_type.lower()
    if road in {"city", "arterial-urban"}:
        return "urban"
    if road in {"arterial-rural", "smaller-rural"}:
        return "rural"
    if road == "highway":
        return "highway"
    return "other"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_base(
    data_dir: Path,
    frame_ids: List[str],
    zod_meta: Dict[str, Dict[str, Any]],
    zod_version: ZodVersion,
) -> Dict[str, Any]:
    """Generate manifest.json by scanning images/annotations and adding ZOD metadata."""
    annotations_dir = data_dir / "annotations"
    entries: List[Dict[str, Any]] = []

    for fid in frame_ids:
        ann_path = annotations_dir / f"{fid}.json"
        if ann_path.exists():
            with ann_path.open() as f:
                ann = json.load(f)
            num_objects = ann.get("num_objects", len(ann.get("annotations", [])))
            categories_present = ann.get("categories_present", [])
        else:
            num_objects = 0
            categories_present = []

        m = zod_meta[fid]
        entries.append({
            "frame_id": fid,
            "image_path": f"images/{fid}.jpg",
            "annotation_path": f"annotations/{fid}.json",
            "num_objects": num_objects,
            "categories_present": categories_present,
            "timestamp": m["timestamp"],
            "split": m["split"],
        })

    ts_count = sum(1 for e in entries if e["timestamp"] is not None)
    print(f"[base] frames: {len(entries)}, with timestamps: {ts_count}")

    return {
        "version": zod_version,
        "resize_width": RESIZE_WIDTH,
        "resize_height": RESIZE_HEIGHT,
        "crop_params": CROP_PARAMS,
        "num_frames": len(entries),
        "category_name_to_id": CATEGORY_NAME_TO_ID,
        "frames": entries,
    }


def build_temporal(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Sort train frames by timestamp."""
    frames = manifest["frames"]
    train = [f for f in frames if f.get("split") == "train"]
    sorted_train = sorted(train, key=_temporal_key)

    out = deepcopy(manifest)
    out["frames"] = _replace_train_order(frames, sorted_train)
    out["ordering"] = {"strategy": "temporal", "split_scope": "train_only"}

    missing = sum(1 for f in train if f.get("timestamp") is None)
    print(f"[temporal] train frames: {len(train)}, missing timestamps: {missing}")
    return out


def build_road_type(
    manifest: Dict[str, Any],
    zod_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Group train frames by road type, temporally sorted within each group."""
    frames = manifest["frames"]
    train = [f for f in frames if f.get("split") == "train"]

    by_road: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in train:
        by_road[zod_meta[f["frame_id"]]["road_type"]].append(f)

    preferred = ["city", "arterial-urban", "highway", "arterial-rural", "smaller-rural"]
    order = [r for r in preferred if r in by_road] + sorted(
        r for r in by_road if r not in preferred
    )

    reordered: List[Dict[str, Any]] = []
    for r in order:
        reordered.extend(sorted(by_road[r], key=_temporal_key))

    out = deepcopy(manifest)
    out["frames"] = _replace_train_order(frames, reordered)
    out["ordering"] = {"strategy": "road_type_blocks", "block_order": order}

    counts = dict(Counter(zod_meta[f["frame_id"]]["road_type"] for f in train))
    seq = [zod_meta[f["frame_id"]]["road_type"] for f in reordered]
    transitions = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    print(f"[road_type] counts: {counts}, transitions: {transitions}")
    return out


def build_urban_rural(
    manifest: Dict[str, Any],
    zod_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Group train frames by urban/rural/highway bucket, temporally sorted within."""
    frames = manifest["frames"]
    train = [f for f in frames if f.get("split") == "train"]

    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in train:
        by_bucket[_urban_rural_bucket(zod_meta[f["frame_id"]]["road_type"])].append(f)

    order = ["urban", "rural", "highway", "other"]
    reordered: List[Dict[str, Any]] = []
    for b in order:
        reordered.extend(sorted(by_bucket.get(b, []), key=_temporal_key))

    out = deepcopy(manifest)
    out["frames"] = _replace_train_order(frames, reordered)
    out["ordering"] = {"strategy": "urban_rural_blocks", "block_order": order}

    counts = dict(Counter(
        _urban_rural_bucket(zod_meta[f["frame_id"]]["road_type"]) for f in train
    ))
    seq = [_urban_rural_bucket(zod_meta[f["frame_id"]]["road_type"]) for f in reordered]
    transitions = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    print(f"[urban_rural] counts: {counts}, transitions: {transitions}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_VARIANTS = ["base", "temporal", "road_type", "urban_rural"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manifest files")
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Preprocessed data directory (contains images/ and annotations/)",
    )
    parser.add_argument(
        "--zod-root", type=str, default=None,
        help="ZOD dataset root (required for base, road_type, urban_rural)",
    )
    parser.add_argument(
        "--zod-version", type=str, default="full", choices=["full", "mini"],
    )
    parser.add_argument(
        "--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS,
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "images").is_dir():
        raise FileNotFoundError(f"No images/ directory in {data_dir}")

    variants = args.variants
    zod_version = cast(ZodVersion, args.zod_version)
    needs_zod = any(v in variants for v in ["base", "road_type", "urban_rural"])
    if needs_zod and not args.zod_root:
        parser.error("--zod-root is required for 'base', 'road_type', and 'urban_rural' variants")

    # Discover frames from image files
    frame_ids = sorted(p.stem for p in (data_dir / "images").glob("*.jpg"))
    print(f"Found {len(frame_ids)} images in {data_dir / 'images'}")

    # Load ZOD metadata once (needed for base + metadata variants)
    zod_meta = None
    if needs_zod:
        zod_meta = _load_zod_metadata(frame_ids, args.zod_root, zod_version)

    # Base manifest
    if "base" in variants:
        if zod_meta is None:
            raise RuntimeError("Internal error: zod metadata not loaded for base variant.")
        manifest = build_base(data_dir, frame_ids, zod_meta, zod_version)
        save_manifest(data_dir / "manifest.json", manifest)
        print(f"[base] wrote: {data_dir / 'manifest.json'}")
    else:
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Run with --variants base first."
            )
        manifest = load_manifest(manifest_path)

    # Ordering variants
    if "temporal" in variants:
        m = build_temporal(manifest)
        save_manifest(data_dir / "manifest_temporal.json", m)
        print(f"[temporal] wrote: {data_dir / 'manifest_temporal.json'}")

    if "road_type" in variants:
        if zod_meta is None:
            raise RuntimeError("Internal error: zod metadata not loaded for road_type variant.")
        m = build_road_type(manifest, zod_meta)
        save_manifest(data_dir / "manifest_road_type.json", m)
        print(f"[road_type] wrote: {data_dir / 'manifest_road_type.json'}")

    if "urban_rural" in variants:
        if zod_meta is None:
            raise RuntimeError("Internal error: zod metadata not loaded for urban_rural variant.")
        m = build_urban_rural(manifest, zod_meta)
        save_manifest(data_dir / "manifest_urban_rural.json", m)
        print(f"[urban_rural] wrote: {data_dir / 'manifest_urban_rural.json'}")


if __name__ == "__main__":
    main()
