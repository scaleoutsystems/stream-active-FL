#!/usr/bin/env python3
"""Generate a temporally-ordered manifest that shares its bootstrap and val
split with manifest_cityday_curated_boot<N>.json.

We reuse the metadata and split assignments already baked into the curated
manifest (bootstrap selection, val split, per-frame timestamps) and only
reorder the streaming portion to be sorted by capture timestamp.  This
isolates "stream order" as the only variable when comparing temporal vs
curated runs at the same bootstrap and val set.

The output manifest declares ordering.strategy = "cityday_temporal_blocks",
which is registered in stream_blocks.py to reuse the cityday_curated block
labeler.  Per-domain validation mAP therefore lives on the same 13 buckets
in both orderings.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


def _temporal_key(frame: Dict[str, Any]) -> str:
    ts = frame.get("timestamp")
    return ts if isinstance(ts, str) else ""


def build_temporal_with_existing_bootstrap(
    curated: Dict[str, Any],
    bootstrap_size: int,
) -> Dict[str, Any]:
    """Take a manifest with bootstrap-prefixed train ordering and replace
    the post-bootstrap train portion with a single temporal block.

    Args:
        curated: Manifest dict whose train split begins with N bootstrap
            frames followed by curated-block-ordered stream frames.
        bootstrap_size: N -- preserved verbatim at the head of train.
    """
    out = deepcopy(curated)
    frames = out["frames"]

    train_idx = [i for i, f in enumerate(frames) if f.get("split") == "train"]
    if len(train_idx) < bootstrap_size:
        raise ValueError(
            f"Train split has {len(train_idx)} frames, "
            f"less than requested bootstrap_size={bootstrap_size}",
        )

    boot_train_idx = train_idx[:bootstrap_size]
    stream_train_idx = train_idx[bootstrap_size:]

    bootstrap_frames = [frames[i] for i in boot_train_idx]
    stream_frames = [frames[i] for i in stream_train_idx]

    missing_ts = sum(1 for f in stream_frames if not _temporal_key(f))
    sorted_stream = sorted(stream_frames, key=_temporal_key)

    new_train = bootstrap_frames + sorted_stream

    new_frames: List[Dict[str, Any]] = []
    cursor = 0
    for f in frames:
        if f.get("split") == "train":
            new_frames.append(new_train[cursor])
            cursor += 1
        else:
            new_frames.append(f)
    out["frames"] = new_frames

    out["ordering"] = {
        "strategy": "cityday_temporal_blocks",
        "bootstrap_frames": bootstrap_size,
        "bootstrap_ids_count": bootstrap_size,
        "stream_frames": len(sorted_stream),
        "stream_order": "temporal",
        "missing_timestamps": missing_ts,
    }
    print(
        f"[cityday_temporal] bootstrap={bootstrap_size}, "
        f"stream={len(sorted_stream)} (missing_ts={missing_ts})"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--curated", type=Path, required=True,
                   help="Path to manifest_cityday_curated_boot<N>.json")
    p.add_argument("--bootstrap-n", type=int, required=True,
                   help="Bootstrap size N (e.g. 2000)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output manifest path")
    args = p.parse_args()

    with args.curated.open("r") as f:
        curated = json.load(f)

    new_manifest = build_temporal_with_existing_bootstrap(
        curated, bootstrap_size=args.bootstrap_n,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(new_manifest, f, indent=2)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
