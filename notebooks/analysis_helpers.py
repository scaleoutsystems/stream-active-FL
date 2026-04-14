"""
Shared helpers for experiment-analysis notebooks.

Provides run discovery, CSV loading, manifest parsing, optional ZOD metadata
enrichment, and forgetting / summary utilities.  Keeps notebook cells thin so
the analysis logic is testable and reusable.

Expected output layout (created by the experiment scripts):

    outputs/
      <pipeline>/            # offline, streaming, federated
        <variant>/           # baseline, no_filter, distribution_filter, ...
          <YYYY-mm-dd_HH-MM-SS>/
            config.yaml
            run_info.json
            epochs.csv             # offline only
            checkpoints.csv        # streaming / federated
            streaming_metrics.csv  # streaming only
            filter_stats.csv       # streaming only
            decisions.csv          # streaming only
            rounds.csv             # federated only
            bootstrap_epochs.csv   # streaming / federated
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# =============================================================================
# Project root
# =============================================================================


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from start until pyproject.toml + src/stream_active_fl exist."""
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists() and (p / "src" / "stream_active_fl").is_dir():
            return p
    raise RuntimeError(
        "Could not locate project root (expected pyproject.toml + src/stream_active_fl)."
    )


# =============================================================================
# Manifest / data-path resolution
# =============================================================================


def resolve_manifest_path(project_root: Path, manifest_path: str | Path) -> Path:
    """Resolve a manifest path the same way the experiment scripts do.

    1. Absolute -> use as-is.
    2. Relative -> try project_root / manifest_path.
    3. If the path starts with data/ and STREAM_ACTIVE_FL_DATA_ROOT is set,
       try that environment-variable override.
    """
    resolved = Path(manifest_path)
    if resolved.is_absolute():
        return resolved
    candidate = project_root / resolved
    if candidate.exists():
        return candidate
    if resolved.parts and resolved.parts[0] == "data":
        data_root = os.environ.get("STREAM_ACTIVE_FL_DATA_ROOT")
        if data_root:
            alt = Path(data_root) / Path(*resolved.parts[1:])
            if alt.exists():
                return alt
    return candidate


# =============================================================================
# Run discovery
# =============================================================================

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def _iter_run_dirs(outputs_root: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (pipeline, variant, run_dir) for every timestamped run directory."""
    if not outputs_root.is_dir():
        return
    for pipeline_dir in sorted(outputs_root.iterdir()):
        if not pipeline_dir.is_dir() or pipeline_dir.name.startswith((".","old")):
            continue
        for variant_dir in sorted(pipeline_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            for run_dir in sorted(variant_dir.iterdir()):
                if run_dir.is_dir() and _TIMESTAMP_RE.match(run_dir.name):
                    yield pipeline_dir.name, variant_dir.name, run_dir


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Load config.yaml from a run directory (empty dict if absent)."""
    p = run_dir / "config.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        result = yaml.safe_load(f)
        return result if isinstance(result, dict) else {}


def load_run_info(run_dir: Path) -> Dict[str, Any]:
    """Load run_info.json from a run directory (empty dict if absent)."""
    p = run_dir / "run_info.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def discover_runs(outputs_root: Path) -> pd.DataFrame:
    """Build a DataFrame with one row per discovered run.

    Columns include pipeline/variant/run_id metadata, boolean flags for
    which CSVs exist, and selected config fields.
    """
    rows: List[Dict[str, Any]] = []
    for pipeline, variant, run_dir in _iter_run_dirs(outputs_root):
        cfg = load_run_config(run_dir)
        info = load_run_info(run_dir)
        row: Dict[str, Any] = {
            "pipeline": pipeline,
            "variant": variant,
            "run_id": run_dir.name,
            "run_dir": run_dir,
            # Which log files are present
            "has_epochs": (run_dir / "epochs.csv").exists(),
            "has_checkpoints": (run_dir / "checkpoints.csv").exists(),
            "has_streaming_metrics": (run_dir / "streaming_metrics.csv").exists(),
            "has_rounds": (run_dir / "rounds.csv").exists(),
            "has_bootstrap_epochs": (run_dir / "bootstrap_epochs.csv").exists(),
            # Config fields
            "seed": cfg.get("seed"),
            "filter_policy": cfg.get("filter_policy"),
            "bootstrap_frames": cfg.get("bootstrap_frames"),
            "bootstrap_epochs": cfg.get("bootstrap_epochs"),
            "buffer_capacity": cfg.get("buffer_capacity"),
            "accept_fraction": cfg.get("accept_fraction"),
            "target_classes": cfg.get("target_classes"),
        }
        if info:
            row["start_time"] = info.get("start_time")
            row["end_time"] = info.get("end_time")
            row["duration_s"] = info.get("duration_seconds")
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["pipeline", "variant", "run_id"]).reset_index(drop=True)


def pick_latest_run(
    runs_df: pd.DataFrame,
    pipeline: str,
    variant: str,
    seed: Optional[int] = None,
) -> Optional[Path]:
    """Return the run directory with the latest timestamp for a pipeline/variant."""
    if runs_df.empty:
        return None
    mask = (runs_df["pipeline"] == pipeline) & (runs_df["variant"] == variant)
    if seed is not None:
        mask = mask & (runs_df["seed"] == seed)
    sub = runs_df.loc[mask]
    if sub.empty:
        return None
    return Path(sub.sort_values("run_id").iloc[-1]["run_dir"])


# =============================================================================
# Variant key lists
# =============================================================================

STREAMING_VARIANTS: List[str] = [
    "no_filter",
    "random_filter",
    "distribution_filter",
    "uncertainty_filter",
    "gradient_norm_filter",
]

FEDERATED_VARIANTS: List[str] = [
    "no_filter",
    "random_filter",
    "distribution_filter",
    "uncertainty_filter",
    "gradient_norm_filter",
]


# =============================================================================
# CSV loading
# =============================================================================


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV file, returning None if it does not exist."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def per_class_ap_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that look like AP_<ClassName>."""
    return [c for c in df.columns if c.startswith("AP_")]


# =============================================================================
# Manifest helpers
# =============================================================================


def load_manifest(project_root: Path, manifest_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a manifest JSON, resolving relative paths."""
    if not manifest_path:
        return None
    p = resolve_manifest_path(project_root, manifest_path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def manifest_to_dataframe(manifest: Dict[str, Any]) -> pd.DataFrame:
    """Convert the frames list inside a manifest to a DataFrame.

    Extracts all useful metadata fields when present.
    """
    _KEYS = [
        "frame_id", "split", "timestamp", "num_objects", "categories_present",
        "scene_bucket", "road_type", "time_of_day", "road_condition",
        "scraped_weather", "country_code", "solar_angle_elevation",
        "num_vehicles", "num_pedestrians", "num_vulnerable_vehicles",
        "num_traffic_signs", "num_traffic_lights",
    ]
    rows = []
    for f in manifest.get("frames", []):
        row: Dict[str, Any] = {}
        for k in _KEYS:
            if k in f:
                row[k] = f[k]
        if "num_objects" not in row:
            row["num_objects"] = 0
        if "categories_present" not in row:
            row["categories_present"] = []
        rows.append(row)
    return pd.DataFrame(rows)


def ordering_summary(manifest: Optional[Dict[str, Any]]) -> str:
    """One-line description of a manifest's ordering strategy."""
    if not manifest:
        return "unknown"
    o = manifest.get("ordering") or {}
    strat = o.get("strategy", "default")
    blocks = o.get("block_order")
    if blocks:
        return f"{strat} | blocks: {blocks}"
    return strat


def block_transitions(
    manifest: Optional[Dict[str, Any]],
    bootstrap_frames: int = 0,
) -> List[Tuple[int, str]]:
    """Return (stream_global_idx, block_label) at each block start.

    Uses ordering.block_sizes (preferred) or falls back to equal-sized
    estimation from ordering.block_order.  bootstrap_frames is subtracted
    so indices align with streaming global_idx.

    For manifests with a shared bootstrap prefix (ordering.bootstrap_frames
    is set), the block_sizes already exclude the bootstrap, so no offset
    is applied.

    Returns an empty list for temporal / default orderings.
    """
    if not manifest:
        return []
    ordering = manifest.get("ordering") or {}
    block_order = ordering.get("block_order")
    if not block_order:
        return []

    train_count = sum(
        1 for f in manifest.get("frames", []) if f.get("split") == "train"
    )
    block_sizes = ordering.get("block_sizes")

    has_shared_bootstrap = ordering.get("bootstrap_frames", 0) > 0
    offset = 0 if has_shared_bootstrap else bootstrap_frames

    transitions: List[Tuple[int, str]] = []
    pos = 0
    for block_label in block_order:
        if block_sizes and block_label in block_sizes:
            size = block_sizes[block_label]
        else:
            size = train_count // len(block_order)
        stream_idx = max(0, pos - offset)
        transitions.append((stream_idx, block_label))
        pos += size
    return transitions


# =============================================================================
# ZOD road-type enrichment
# =============================================================================


def urban_rural_bucket(road_type: str) -> str:
    """Map a ZOD road_type string to a coarse scene bucket."""
    road = str(road_type).lower()
    if road in {"city", "arterial-urban"}:
        return "urban"
    if road in {"arterial-rural", "smaller-rural"}:
        return "rural"
    if road == "highway":
        return "highway"
    return "other"


def try_load_road_types(
    frame_ids: Sequence[str],
    zod_root: Optional[str],
    zod_version: str = "full",
) -> Optional[pd.DataFrame]:
    """Return a frame_id -> road_type, scene_bucket DataFrame via the zod package.

    Returns None when zod_root is unset, the directory is missing, or
    the zod package is not importable.
    """
    if not zod_root or not Path(zod_root).is_dir():
        return None
    try:
        from zod import ZodFrames  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        zf = ZodFrames(zod_root, zod_version)  # type: ignore[arg-type]
    except Exception:
        return None

    rows = []
    for fid in frame_ids:
        try:
            rt = str(getattr(zf[fid].metadata, "road_type", "unknown") or "unknown")
        except Exception:
            rt = "unknown"
        rows.append({"frame_id": fid, "road_type": rt, "scene_bucket": urban_rural_bucket(rt)})
    return pd.DataFrame(rows)


# =============================================================================
# Decisions enrichment
# =============================================================================


def parse_categories_cell(x: Any) -> List[str]:
    """Parse a semicolon-separated category string from decisions.csv."""
    if pd.isna(x) or x == "":
        return []
    return [c for c in str(x).split(";") if c]


def enrich_decisions(
    decisions: pd.DataFrame,
    frames_df: pd.DataFrame,
    road_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge decisions.csv with manifest frames and optional road-type metadata."""
    d = decisions.copy()
    d["categories_list"] = d["categories"].map(parse_categories_cell)
    d["n_categories"] = d["categories_list"].map(len)
    d["empty_scene"] = d["n_categories"] == 0
    if "frame_id" not in d.columns:
        return d
    d["frame_id"] = d["frame_id"].astype(str).str.zfill(6)
    frames_df = frames_df.copy()
    frames_df["frame_id"] = frames_df["frame_id"].astype(str).str.zfill(6)
    keep = [
        "frame_id", "timestamp", "split", "num_objects", "scene_bucket",
        "road_type", "time_of_day", "road_condition", "scraped_weather",
        "country_code", "num_vehicles", "num_pedestrians",
        "num_vulnerable_vehicles", "num_traffic_signs", "num_traffic_lights",
        "solar_angle_elevation",
    ]
    keep = [c for c in keep if c in frames_df.columns]
    out = d.merge(frames_df[keep], on="frame_id", how="left", suffixes=("", "_manifest"))
    if road_df is not None:
        rd = road_df.copy()
        rd["frame_id"] = rd["frame_id"].astype(str).str.zfill(6)
        out = out.merge(rd, on="frame_id", how="left")
    return out


# =============================================================================
# Rolling statistics
# =============================================================================


def rolling_accept_rate(
    enriched: pd.DataFrame,
    bucket_col: str,
    window: int = 500,
) -> Optional[pd.DataFrame]:
    """Compute rolling accept rate within each bucket along global_idx."""
    if bucket_col not in enriched.columns:
        return None
    parts = []
    for bucket, grp in enriched.groupby(bucket_col, dropna=False):
        grp = grp.sort_values("global_idx")
        win = min(window, max(5, len(grp) // 10))
        acc = grp["action"].eq("accept").astype(float)
        parts.append(
            pd.DataFrame({
                bucket_col: bucket,
                "global_idx": grp["global_idx"].values,
                "roll_accept_rate": acc.rolling(win, min_periods=1).mean().values,
            })
        )
    return pd.concat(parts, ignore_index=True) if parts else None


def rolling_class_presence(
    decisions: pd.DataFrame,
    class_name: str,
    action: str,
    window: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling fraction of frames containing class_name for a given action.

    Returns (global_idx, rolling_fraction) arrays.
    """
    sub = decisions.loc[decisions["action"] == action].sort_values("global_idx")
    if sub.empty:
        return np.array([]), np.array([])
    hits = sub["categories"].fillna("").apply(lambda s: class_name in s.split(";"))
    win = min(window, max(20, len(sub) // 15))
    y = hits.astype(float).rolling(win, min_periods=1).mean().to_numpy()
    return sub["global_idx"].to_numpy(), y


# =============================================================================
# Forgetting proxy
# =============================================================================


def forgetting_table(
    checkpoints: pd.DataFrame,
    ap_cols: List[str],
    n_bins: int = 4,
) -> pd.DataFrame:
    """Compare mean per-class AP in the first vs last stream quartile.

    A negative delta indicates the model got worse on that class over the
    stream -- a proxy for catastrophic forgetting or distribution shift.
    """
    if checkpoints.empty or "items_processed" not in checkpoints.columns:
        return pd.DataFrame()
    x = checkpoints.sort_values("items_processed")
    bins = pd.qcut(x["items_processed"], q=min(n_bins, len(x)), duplicates="drop")
    x = x.assign(_bin=bins)
    parts = []
    for label in sorted(x["_bin"].dropna().unique(), key=str):
        parts.append(x.loc[x["_bin"] == label, ap_cols].mean().rename(str(label)))
    if len(parts) < 2:
        return pd.DataFrame()
    out = pd.DataFrame({"early": parts[0], "late": parts[-1]})
    out["delta"] = out["late"] - out["early"]
    return out


# =============================================================================
# Summary table builder
# =============================================================================


def load_enriched_streaming_decisions(
    run_dir: Path,
    project_root: Path,
    *,
    zod_root: Optional[str] = None,
    zod_version: str = "full",
) -> pd.DataFrame:
    """Load decisions.csv merged with manifest frames and optional ZOD road type.

    Adds ts_parsed (UTC) and hours_from_start when timestamps parse.
    """
    dec = read_csv(run_dir / "decisions.csv")
    if dec is None or dec.empty:
        return pd.DataFrame()
    cfg = load_run_config(run_dir)
    man = load_manifest(project_root, str(cfg.get("manifest_path", "")))
    if not man:
        out = dec.copy()
        if "frame_id" in out.columns:
            out["frame_id"] = out["frame_id"].astype(str).str.zfill(6)
        return out
    frames_df = manifest_to_dataframe(man)
    road_df: Optional[pd.DataFrame] = None
    if zod_root and "frame_id" in dec.columns:
        fids_raw = dec["frame_id"].astype(str).str.zfill(6).unique().tolist()
        road_df = try_load_road_types(fids_raw, zod_root, zod_version)
    out = enrich_decisions(dec, frames_df, road_df)
    if "timestamp" in out.columns:
        out["ts_parsed"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        t0 = out["ts_parsed"].min()
        out["hours_from_start"] = (out["ts_parsed"] - t0).dt.total_seconds() / 3600.0
    return out


def subsample_decisions(
    df: pd.DataFrame,
    n_max: int,
    seed: int = 0,
    *,
    stratify_col: str = "action",
) -> pd.DataFrame:
    """Random subsample up to n_max rows, optionally stratified by stratify_col."""
    if df.empty or len(df) <= n_max:
        return df.copy()
    rng = np.random.RandomState(seed)
    if stratify_col in df.columns:
        parts: List[pd.DataFrame] = []
        for _, grp in df.groupby(stratify_col):
            k = max(1, int(round(n_max * len(grp) / len(df))))
            k = min(k, len(grp))
            parts.append(grp.sample(n=k, random_state=rng))
        out = pd.concat(parts, ignore_index=True)
        if len(out) > n_max:
            out = out.sample(n=n_max, random_state=rng)
        return out.reset_index(drop=True)
    return df.sample(n=n_max, random_state=rng).reset_index(drop=True)


def inter_accept_gaps(decisions: pd.DataFrame) -> pd.Series:
    """Return gaps in global_idx between consecutive accepted frames."""
    if decisions.empty or "global_idx" not in decisions.columns:
        return pd.Series(dtype=float)
    acc = decisions.loc[decisions["action"] == "accept"].sort_values("global_idx")
    return acc["global_idx"].diff().dropna()


def pca_two_components(X: np.ndarray) -> np.ndarray:
    """Project X (n x d) onto the first two principal components (NumPy SVD)."""
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0 or X.shape[0] < 3:
        return np.zeros((X.shape[0], 2))
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def build_summary_row(
    label: str,
    run_dir: Path,
) -> Dict[str, Any]:
    """Extract best/last mAP from whichever CSV is available in run_dir."""
    cfg = load_run_config(run_dir)
    info = load_run_info(run_dir)
    row: Dict[str, Any] = {
        "experiment": label,
        "run_id": run_dir.name,
        "seed": cfg.get("seed"),
        "filter_policy": cfg.get("filter_policy"),
        "duration_h": round(info.get("duration_seconds", 0) / 3600, 2) if info.get("duration_seconds") else None,
    }

    # Streaming checkpoints
    ck = read_csv(run_dir / "checkpoints.csv")
    if ck is not None and not ck.empty and "mAP" in ck.columns:
        idx = ck["mAP"].idxmax()
        row["best_mAP"] = ck.loc[idx, "mAP"]
        row["best_at_items"] = ck.loc[idx, "items_processed"]
        row["last_mAP"] = ck["mAP"].iloc[-1]

    # Offline epochs
    ep = read_csv(run_dir / "epochs.csv")
    if ep is not None and not ep.empty and "mAP" in ep.columns:
        valid = ep.dropna(subset=["mAP"])
        if not valid.empty:
            idx = valid["mAP"].idxmax()
            row["best_mAP"] = valid.loc[idx, "mAP"]
            row["best_epoch"] = valid.loc[idx, "epoch"]
            row["last_mAP"] = valid["mAP"].iloc[-1]

    # Federated rounds
    rd = read_csv(run_dir / "rounds.csv")
    if rd is not None and not rd.empty and "mAP" in rd.columns:
        idx = rd["mAP"].idxmax()
        row["best_mAP"] = rd.loc[idx, "mAP"]
        row["best_round"] = rd.loc[idx, "round"]
        row["last_mAP"] = rd["mAP"].iloc[-1]

    # best_stream_mAP from run_info (top-level key)
    if "best_stream_mAP" in info:
        row["best_stream_mAP"] = info["best_stream_mAP"]

    return row
