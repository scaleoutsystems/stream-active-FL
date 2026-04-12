#!/usr/bin/env python3
"""
Generate manifest files for streaming experiments from preprocessed ZOD data.

The base manifest is built by scanning a preprocessed data directory
(images/ and annotations/) and looking up ZOD metadata for each
frame (timestamps, train/val splits, road type, weather, etc.).

Ordering variants reorder the train frames into contiguous domain blocks
while keeping val frames in their original positions.

Outputs (all written to <data-dir>/):

  manifest.json                  Base manifest (original frame order)
  manifest_temporal.json         Train frames sorted by capture timestamp
  manifest_road_type.json        Blocks by ZOD road_type
                                 (city | arterial-urban | highway |
                                 arterial-rural | smaller-rural)
  manifest_road_type_reverse.json  Same blocks in reverse order
  manifest_road_type_time.json   Compound blocks: road_type x time_of_day
  manifest_conditions.json       Blocks by weather / road-surface condition
  manifest_urban_rural.json      Coarse 3-way grouping (urban | rural | highway)

Usage:

    # Generate all variants (requires ZOD access):
    python tools/preprocessing/build_manifests.py \\
        --data-dir data/Frames_1600x480 --zod-root /path/to/zod

    # Regenerate only ordering variants (base manifest must exist):
    python tools/preprocessing/build_manifests.py \\
        --data-dir data/Frames_1600x480 --zod-root /path/to/zod \\
        --variants road_type road_type_reverse conditions
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, cast

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CATEGORY_NAME_TO_ID,
    CROP_PARAMS,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    extract_timestamp,
)

ZodVersion = Literal["full", "mini"]
ZodMeta = Dict[str, Dict[str, Any]]
BucketFn = Callable[[Dict[str, Any]], str]


# =============================================================================
# IO helpers
# =============================================================================


def load_manifest(path: Path) -> Dict[str, Any]:
    """Read a manifest JSON file."""
    with path.open("r") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Write a manifest dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)


# =============================================================================
# ZOD metadata extraction
# =============================================================================

_ZOD_STR_FIELDS = ("road_type", "time_of_day", "road_condition",
                    "scraped_weather", "country_code")
_ZOD_INT_FIELDS = ("num_vehicles", "num_pedestrians",
                    "num_vulnerable_vehicles", "num_traffic_signs",
                    "num_traffic_lights")


def _load_zod_metadata(
    frame_ids: List[str],
    zod_root: str,
    zod_version: ZodVersion,
) -> ZodMeta:
    """Load per-frame metadata from the ZOD dataset.

    Returns a dict mapping frame_id to a metadata dict containing:
    timestamp, split, road_type, time_of_day, road_condition,
    scraped_weather, country_code, num_vehicles, num_pedestrians,
    num_vulnerable_vehicles, num_traffic_signs, num_traffic_lights,
    solar_angle_elevation.
    """
    from zod import ZodFrames
    from zod.constants import TRAIN, VAL

    zod_frames = ZodFrames(zod_root, zod_version)
    train_ids = zod_frames.get_split(TRAIN)
    val_ids = zod_frames.get_split(VAL)

    meta: ZodMeta = {}
    for fid in tqdm(frame_ids, desc="Loading ZOD metadata"):
        fm = zod_frames[fid].metadata

        entry: Dict[str, Any] = {
            "timestamp": extract_timestamp(fm),
            "split": (
                "train" if fid in train_ids
                else ("val" if fid in val_ids else "unknown")
            ),
            "solar_angle_elevation": float(
                getattr(fm, "solar_angle_elevation", 0.0) or 0.0
            ),
        }
        for field in _ZOD_STR_FIELDS:
            entry[field] = str(getattr(fm, field, "unknown") or "unknown")
        for field in _ZOD_INT_FIELDS:
            entry[field] = int(getattr(fm, field, 0) or 0)

        meta[fid] = entry

    return meta


# =============================================================================
# Internal helpers
# =============================================================================


def _temporal_key(frame: Dict[str, Any]) -> Tuple[int, str, str]:
    """Sort key: frames with timestamps first, then by timestamp, then id."""
    ts = frame.get("timestamp")
    if ts is None:
        return (1, "", frame.get("frame_id", ""))
    return (0, str(ts), frame.get("frame_id", ""))


def _replace_train_order(
    all_frames: List[Dict[str, Any]],
    reordered_train: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Replace train frames in all_frames with reordered_train, keeping
    val frames in their original positions."""
    idx = 0
    out: List[Dict[str, Any]] = []
    for f in all_frames:
        if f.get("split") == "train":
            out.append(reordered_train[idx])
            idx += 1
        else:
            out.append(f)
    if idx != len(reordered_train):
        raise ValueError(
            f"Train frame count mismatch: expected {len(reordered_train)}, "
            f"placed {idx}"
        )
    return out


def _urban_rural_bucket(road_type: str) -> str:
    """Map a ZOD road_type string to a coarse 3-way bucket."""
    road = road_type.lower()
    if road in {"city", "arterial-urban"}:
        return "urban"
    if road in {"arterial-rural", "smaller-rural"}:
        return "rural"
    if road == "highway":
        return "highway"
    return "other"


def _weather_bucket(weather: str, road_condition: str) -> str:
    """Map scraped_weather + road_condition to a coarse conditions bucket."""
    w = weather.lower()
    rc = road_condition.lower()
    if "snow" in w or "snow" in rc:
        return "snow"
    if "rain" in w or "wet" in rc:
        return "rain_wet"
    if "fog" in w:
        return "fog"
    if "cloud" in w or "overcast" in w:
        return "cloudy"
    return "clear"


def _enrich_frame(entry: Dict[str, Any], zod_meta: ZodMeta) -> None:
    """Copy ZOD metadata fields into a frame entry dict (in-place)."""
    m = zod_meta[entry["frame_id"]]
    for field in _ZOD_STR_FIELDS:
        entry[field] = m[field]
    for field in _ZOD_INT_FIELDS:
        entry[field] = m[field]
    entry["solar_angle_elevation"] = m["solar_angle_elevation"]


# =============================================================================
# Manifest builders
# =============================================================================


def build_base(
    data_dir: Path,
    frame_ids: List[str],
    zod_meta: ZodMeta,
    zod_version: ZodVersion,
) -> Dict[str, Any]:
    """Build the base manifest by scanning images/annotations and merging
    ZOD metadata.

    Each frame entry contains annotation info (num_objects,
    categories_present) plus all available ZOD metadata fields.
    """
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
        entry: Dict[str, Any] = {
            "frame_id": fid,
            "image_path": f"images/{fid}.jpg",
            "annotation_path": f"annotations/{fid}.json",
            "num_objects": num_objects,
            "categories_present": categories_present,
            "timestamp": m["timestamp"],
            "split": m["split"],
        }
        _enrich_frame(entry, zod_meta)
        entries.append(entry)

    ts_count = sum(1 for e in entries if e["timestamp"] is not None)
    train = [e for e in entries if e["split"] == "train"]
    print(f"[base] {len(entries)} frames ({len(train)} train), "
          f"{ts_count} with timestamps")
    for field in ("road_type", "time_of_day", "road_condition", "scraped_weather"):
        dist = Counter(e.get(field, "unknown") for e in train)
        print(f"  {field}: {dict(dist)}")

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
    """Reorder train frames by capture timestamp (ascending)."""
    frames = manifest["frames"]
    train = [f for f in frames if f.get("split") == "train"]
    sorted_train = sorted(train, key=_temporal_key)

    out = deepcopy(manifest)
    out["frames"] = _replace_train_order(out["frames"], sorted_train)
    out["ordering"] = {"strategy": "temporal", "split_scope": "train_only"}

    missing = sum(1 for f in train if f.get("timestamp") is None)
    print(f"[temporal] {len(train)} train frames, {missing} missing timestamps")
    return out


def _build_block_manifest(
    manifest: Dict[str, Any],
    block_order: List[str],
    bucket_fn: BucketFn,
    strategy_name: str,
    bucket_field: str = "scene_bucket",
) -> Dict[str, Any]:
    """Group train frames into contiguous blocks, temporally sorted within
    each block.

    Args:
        manifest: Source manifest (not modified).
        block_order: Preferred ordering of block labels.  Blocks present
            in the data appear in this order; any extra blocks are appended
            alphabetically.
        bucket_fn: Maps a frame dict to its block label string.
        strategy_name: Value stored in ordering.strategy.
        bucket_field: Key added to each train frame with its block label.

    Returns:
        A new manifest dict with reordered train frames.
    """
    out = deepcopy(manifest)
    train = [f for f in out["frames"] if f.get("split") == "train"]

    by_block: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in train:
        bucket = bucket_fn(f)
        f[bucket_field] = bucket
        by_block[bucket].append(f)

    order = [b for b in block_order if b in by_block] + sorted(
        b for b in by_block if b not in block_order
    )

    reordered: List[Dict[str, Any]] = []
    for b in order:
        reordered.extend(sorted(by_block[b], key=_temporal_key))

    out["frames"] = _replace_train_order(out["frames"], reordered)

    block_sizes = {b: len(by_block[b]) for b in order}
    out["ordering"] = {
        "strategy": strategy_name,
        "block_order": order,
        "block_sizes": block_sizes,
    }

    print(f"[{strategy_name}] {block_sizes}")
    return out


# ---------------------------------------------------------------------------
# Road-type orderings
# ---------------------------------------------------------------------------

ROAD_TYPE_ORDER = [
    "city", "arterial-urban", "highway", "arterial-rural", "smaller-rural",
]


def build_road_type(
    manifest: Dict[str, Any],
    zod_meta: ZodMeta,
) -> Dict[str, Any]:
    """Group train frames by ZOD road_type.

    Block order: city, arterial-urban, highway, arterial-rural, smaller-rural.
    """
    return _build_block_manifest(
        manifest,
        block_order=ROAD_TYPE_ORDER,
        bucket_fn=lambda f: zod_meta[f["frame_id"]]["road_type"],
        strategy_name="road_type_blocks",
    )


def build_road_type_reverse(
    manifest: Dict[str, Any],
    zod_meta: ZodMeta,
) -> Dict[str, Any]:
    """Group train frames by ZOD road_type in reverse order.

    Block order: smaller-rural, arterial-rural, highway, arterial-urban, city.
    """
    return _build_block_manifest(
        manifest,
        block_order=list(reversed(ROAD_TYPE_ORDER)),
        bucket_fn=lambda f: zod_meta[f["frame_id"]]["road_type"],
        strategy_name="road_type_reverse_blocks",
    )


# ---------------------------------------------------------------------------
# Compound orderings
# ---------------------------------------------------------------------------


def build_road_type_time(
    manifest: Dict[str, Any],
    zod_meta: ZodMeta,
) -> Dict[str, Any]:
    """Group train frames by (road_type, time_of_day) compound blocks.

    Creates blocks like city_day, city_night, highway_day, etc.
    Road types appear in standard order; within each, day before twilight
    before night.
    """
    time_order = ["day", "twilight", "night"]
    preferred = [f"{r}_{t}" for r in ROAD_TYPE_ORDER for t in time_order]

    def bucket_fn(f: Dict[str, Any]) -> str:
        m = zod_meta[f["frame_id"]]
        return f"{m['road_type']}_{m['time_of_day']}"

    return _build_block_manifest(
        manifest,
        block_order=preferred,
        bucket_fn=bucket_fn,
        strategy_name="road_type_time_blocks",
    )


def build_conditions(
    manifest: Dict[str, Any],
    zod_meta: ZodMeta,
) -> Dict[str, Any]:
    """Group train frames by weather / road-surface condition.

    Buckets: clear, cloudy, fog, rain_wet, snow.
    """
    preferred = ["clear", "cloudy", "fog", "rain_wet", "snow"]

    def bucket_fn(f: Dict[str, Any]) -> str:
        m = zod_meta[f["frame_id"]]
        return _weather_bucket(m["scraped_weather"], m["road_condition"])

    return _build_block_manifest(
        manifest,
        block_order=preferred,
        bucket_fn=bucket_fn,
        strategy_name="conditions_blocks",
    )


def build_urban_rural(
    manifest: Dict[str, Any],
    zod_meta: ZodMeta,
) -> Dict[str, Any]:
    """Group train frames into coarse urban / rural / highway blocks."""
    preferred = ["urban", "rural", "highway", "other"]

    def bucket_fn(f: Dict[str, Any]) -> str:
        return _urban_rural_bucket(zod_meta[f["frame_id"]]["road_type"])

    return _build_block_manifest(
        manifest,
        block_order=preferred,
        bucket_fn=bucket_fn,
        strategy_name="urban_rural_blocks",
    )


# =============================================================================
# Bootstrap ID generation
# =============================================================================


def build_bootstrap_set(
    manifest: Dict[str, Any],
    *,
    road_type: str = "city",
    time_of_day: str | None = None,
    n: int = 5000,
) -> List[str]:
    """Select n train frame IDs matching the given criteria.

    Frames are filtered by road_type (required) and optionally
    time_of_day.  When time_of_day is None, a proportional
    mix of day/night/twilight is drawn for the selected road type.

    Within each stratum, frames are sorted by timestamp so the
    selection is deterministic.
    """
    train = [f for f in manifest["frames"] if f.get("split") == "train"]
    matching = [f for f in train if f.get("road_type") == road_type]

    if time_of_day is not None:
        matching = [f for f in matching if f.get("time_of_day") == time_of_day]
        matching.sort(key=_temporal_key)
        if len(matching) < n:
            raise ValueError(
                f"Only {len(matching)} frames match road_type={road_type}, "
                f"time_of_day={time_of_day} (need {n})"
            )
        selected = matching[:n]
    else:
        by_tod: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for f in matching:
            by_tod[f.get("time_of_day", "unknown")].append(f)
        for v in by_tod.values():
            v.sort(key=_temporal_key)

        total = len(matching)
        selected_frames: List[Dict[str, Any]] = []
        remaining = n
        tod_keys = sorted(by_tod.keys())
        for i, tod in enumerate(tod_keys):
            if i == len(tod_keys) - 1:
                count = remaining
            else:
                count = round(n * len(by_tod[tod]) / total)
                count = min(count, remaining, len(by_tod[tod]))
            selected_frames.extend(by_tod[tod][:count])
            remaining -= count
        if len(selected_frames) < n:
            for tod in tod_keys:
                for f in by_tod[tod]:
                    if f not in selected_frames:
                        selected_frames.append(f)
                        if len(selected_frames) >= n:
                            break
                if len(selected_frames) >= n:
                    break
        selected = selected_frames[:n]

    ids = [f["frame_id"] for f in selected]

    tod_dist = Counter(f.get("time_of_day") for f in selected)
    ped_count = sum(
        1 for f in selected
        if "Pedestrian" in (f.get("categories_present") or [])
    )
    print(f"[bootstrap] {len(ids)} frames: road_type={road_type}, "
          f"time_of_day={time_of_day or 'proportional'}")
    print(f"  time_of_day: {dict(tod_dist)}")
    print(f"  Pedestrian presence: {ped_count}/{len(ids)} "
          f"({ped_count / len(ids) * 100:.1f}%)")
    return ids


def _build_block_manifest_with_bootstrap(
    manifest: Dict[str, Any],
    block_order: List[str],
    bucket_fn: BucketFn,
    strategy_name: str,
    bootstrap_ids: List[str],
    bucket_field: str = "scene_bucket",
) -> Dict[str, Any]:
    """Like _build_block_manifest but places bootstrap_ids at the start
    of the train split and excludes them from the block-ordered stream
    portion.
    """
    out = deepcopy(manifest)
    train = [f for f in out["frames"] if f.get("split") == "train"]

    boot_set = set(bootstrap_ids)
    boot_by_id = {f["frame_id"]: f for f in train if f["frame_id"] in boot_set}
    stream_frames = [f for f in train if f["frame_id"] not in boot_set]

    bootstrap_ordered = [boot_by_id[fid] for fid in bootstrap_ids
                         if fid in boot_by_id]
    for f in bootstrap_ordered:
        f[bucket_field] = f.get("road_type", "bootstrap")

    by_block: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in stream_frames:
        bucket = bucket_fn(f)
        f[bucket_field] = bucket
        by_block[bucket].append(f)

    order = [b for b in block_order if b in by_block] + sorted(
        b for b in by_block if b not in block_order
    )

    reordered: List[Dict[str, Any]] = []
    reordered.extend(bootstrap_ordered)
    for b in order:
        reordered.extend(sorted(by_block[b], key=_temporal_key))

    out["frames"] = _replace_train_order(out["frames"], reordered)

    block_sizes = {b: len(by_block[b]) for b in order}
    out["ordering"] = {
        "strategy": strategy_name,
        "block_order": order,
        "block_sizes": block_sizes,
        "bootstrap_frames": len(bootstrap_ordered),
        "bootstrap_ids_count": len(bootstrap_ordered),
    }

    print(f"[{strategy_name}] bootstrap={len(bootstrap_ordered)}, "
          f"stream blocks={block_sizes}")
    return out


# =============================================================================
# CLI
# =============================================================================

ALL_VARIANTS = [
    "base", "temporal", "road_type", "road_type_reverse",
    "road_type_time", "conditions", "urban_rural",
]

_VARIANTS_NEEDING_ZOD = {
    "base", "road_type", "road_type_reverse", "road_type_time",
    "conditions", "urban_rural",
}

_VARIANT_FILENAMES: Dict[str, str] = {
    "temporal": "manifest_temporal.json",
    "road_type": "manifest_road_type.json",
    "road_type_reverse": "manifest_road_type_reverse.json",
    "road_type_time": "manifest_road_type_time.json",
    "conditions": "manifest_conditions.json",
    "urban_rural": "manifest_urban_rural.json",
}


_BOOTSTRAP_PRESETS: Dict[str, Dict[str, Any]] = {
    "city_day": {"road_type": "city", "time_of_day": "day", "n": 5000},
    "city_mixed": {"road_type": "city", "time_of_day": None, "n": 5000},
}

_BOOTSTRAP_ORDERING_COMBOS: List[Tuple[str, str, str]] = [
    # (bootstrap_preset, ordering_variant, output_filename)
    ("city_day", "road_type", "manifest_cityday_road_type.json"),
    ("city_day", "road_type_reverse", "manifest_cityday_reverse.json"),
    ("city_mixed", "road_type", "manifest_citymix_road_type.json"),
    ("city_mixed", "road_type_reverse", "manifest_citymix_reverse.json"),
    ("city_mixed", "conditions", "manifest_citymix_conditions.json"),
]


def _get_ordering_params(
    variant: str, zod_meta: ZodMeta,
) -> Tuple[List[str], BucketFn, str]:
    """Return (block_order, bucket_fn, strategy_name) for a given ordering."""
    if variant == "road_type":
        return (
            ROAD_TYPE_ORDER,
            lambda f: zod_meta[f["frame_id"]]["road_type"],
            "road_type_blocks",
        )
    elif variant == "road_type_reverse":
        return (
            list(reversed(ROAD_TYPE_ORDER)),
            lambda f: zod_meta[f["frame_id"]]["road_type"],
            "road_type_reverse_blocks",
        )
    elif variant == "conditions":
        return (
            ["clear", "cloudy", "fog", "rain_wet", "snow"],
            lambda f: _weather_bucket(
                zod_meta[f["frame_id"]]["scraped_weather"],
                zod_meta[f["frame_id"]]["road_condition"],
            ),
            "conditions_blocks",
        )
    else:
        raise ValueError(f"Unknown ordering variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manifest files for streaming experiments",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Preprocessed data directory (contains images/ and annotations/)",
    )
    parser.add_argument(
        "--zod-root", type=str, default=None,
        help="ZOD dataset root (required for most variants)",
    )
    parser.add_argument(
        "--zod-version", type=str, default="full", choices=["full", "mini"],
    )
    parser.add_argument(
        "--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS,
    )
    parser.add_argument(
        "--generate-bootstraps", action="store_true",
        help="Generate bootstrap ID files (city_day and city_mixed)",
    )
    parser.add_argument(
        "--generate-shared-manifests", action="store_true",
        help="Generate bootstrap-prefixed manifests for all combos",
    )
    parser.add_argument(
        "--bootstrap-ids", type=str, default=None,
        help="Path to a bootstrap IDs JSON file (for single manifest gen)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "images").is_dir():
        raise FileNotFoundError(f"No images/ directory in {data_dir}")

    variants: List[str] = args.variants
    zod_version = cast(ZodVersion, args.zod_version)

    needs_zod = bool(set(variants) & _VARIANTS_NEEDING_ZOD)
    if args.generate_bootstraps or args.generate_shared_manifests:
        needs_zod = True
    if needs_zod and not args.zod_root:
        needed = sorted(set(variants) & _VARIANTS_NEEDING_ZOD)
        parser.error(
            f"--zod-root is required for variants: {', '.join(needed)}"
            if needed else "--zod-root is required for bootstrap/shared manifests"
        )

    frame_ids = sorted(p.stem for p in (data_dir / "images").glob("*.jpg"))
    print(f"Found {len(frame_ids)} images in {data_dir / 'images'}")

    zod_meta: ZodMeta | None = None
    if needs_zod:
        assert args.zod_root is not None
        zod_meta = _load_zod_metadata(frame_ids, args.zod_root, zod_version)

    # -- Base manifest --------------------------------------------------------

    if "base" in variants:
        assert zod_meta is not None
        manifest = build_base(data_dir, frame_ids, zod_meta, zod_version)
        save_manifest(data_dir / "manifest.json", manifest)
        print(f"  -> {data_dir / 'manifest.json'}")
    else:
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found.  Run with --variants base first."
            )
        manifest = load_manifest(manifest_path)

    # -- Ordering variants ----------------------------------------------------

    builders: Dict[str, Callable[[], Dict[str, Any]]] = {
        "temporal": lambda: build_temporal(manifest),
    }
    if zod_meta is not None:
        zm = zod_meta
        builders.update({
            "road_type": lambda: build_road_type(manifest, zm),
            "road_type_reverse": lambda: build_road_type_reverse(manifest, zm),
            "road_type_time": lambda: build_road_type_time(manifest, zm),
            "conditions": lambda: build_conditions(manifest, zm),
            "urban_rural": lambda: build_urban_rural(manifest, zm),
        })

    for name in variants:
        if name == "base" or name not in builders:
            continue
        filename = _VARIANT_FILENAMES[name]
        m = builders[name]()
        save_manifest(data_dir / filename, m)
        print(f"  -> {data_dir / filename}")

    # -- Bootstrap ID generation ----------------------------------------------

    if args.generate_bootstraps:
        for preset_name, preset_kwargs in _BOOTSTRAP_PRESETS.items():
            ids = build_bootstrap_set(manifest, **preset_kwargs)
            out_path = data_dir / f"bootstrap_{preset_name}.json"
            with out_path.open("w") as f:
                json.dump(ids, f, indent=2)
            print(f"  -> {out_path} ({len(ids)} IDs)")

    # -- Bootstrap-prefixed manifests -----------------------------------------

    if args.generate_shared_manifests:
        assert zod_meta is not None
        zm = zod_meta
        for boot_preset, ordering, out_name in _BOOTSTRAP_ORDERING_COMBOS:
            boot_path = data_dir / f"bootstrap_{boot_preset}.json"
            if not boot_path.exists():
                print(f"  [SKIP] {boot_path} not found -- "
                      f"run with --generate-bootstraps first")
                continue
            with boot_path.open("r") as f:
                boot_ids: List[str] = json.load(f)

            block_order, bucket_fn, strategy_name = _get_ordering_params(
                ordering, zm,
            )
            m = _build_block_manifest_with_bootstrap(
                manifest,
                block_order=block_order,
                bucket_fn=bucket_fn,
                strategy_name=strategy_name,
                bootstrap_ids=boot_ids,
            )
            save_manifest(data_dir / out_name, m)
            print(f"  -> {data_dir / out_name}")


if __name__ == "__main__":
    main()
