"""
Generic primitives for analyzing experiment outputs.

Provides run discovery, CSV loading, manifest parsing, optional ZOD
metadata enrichment, and forgetting / summary utilities.  The
streaming-specific and federated-specific submodules build on top.

Expected output layout (created by the experiment scripts):

    outputs/
      <pipeline>/            # streaming or federated
        <variant>/
          seed_<N>/
            <YYYY-mm-dd_HH-MM-SS>/
              config.yaml
              run_info.json
              checkpoints.csv        # streaming only
              streaming_metrics.csv  # streaming only
              filter_stats.csv       # streaming only
              decisions.csv          # streaming + federated
              rounds.csv             # federated only
              refreshes.csv          # adaptive runs
              bootstrap_epochs.csv
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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


_SEED_RE = re.compile(r"^seed_(\d+)$")


def _iter_run_dirs(
    outputs_root: Path,
) -> Iterable[Tuple[str, str, Optional[int], Path]]:
    """Yield (pipeline, variant, seed, run_dir) for every timestamped run.

    Supports both legacy layout ('variant/<timestamp>/') and the multi-seed
    layout ('variant/seed_<N>/<timestamp>/').  The variant name is the same
    in both layouts; the seed under 'seed_<N>/' is carried separately so
    seed filtering works uniformly across all runs.
    """
    if not outputs_root.is_dir():
        return
    for pipeline_dir in sorted(outputs_root.iterdir()):
        if not pipeline_dir.is_dir() or pipeline_dir.name.startswith((".", "old")):
            continue
        for variant_dir in sorted(pipeline_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            for entry in sorted(variant_dir.iterdir()):
                if not entry.is_dir():
                    continue
                if _TIMESTAMP_RE.match(entry.name):
                    yield pipeline_dir.name, variant_dir.name, None, entry
                    continue
                seed_match = _SEED_RE.match(entry.name)
                if seed_match:
                    seed = int(seed_match.group(1))
                    for run_dir in sorted(entry.iterdir()):
                        if run_dir.is_dir() and _TIMESTAMP_RE.match(run_dir.name):
                            yield pipeline_dir.name, variant_dir.name, seed, run_dir


def load_run_config(run_dir: Optional[Path]) -> Dict[str, Any]:
    """Load config.yaml from a run directory (empty dict if absent or None)."""
    if run_dir is None:
        return {}
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
    for pipeline, variant, seed_from_path, run_dir in _iter_run_dirs(outputs_root):
        cfg = load_run_config(run_dir)
        info = load_run_info(run_dir)
        # Path-derived seed (from seed_<N>/ layout) overrides config-derived seed;
        # this matches the directory structure the experiment scripts use.
        seed = seed_from_path if seed_from_path is not None else cfg.get("seed")
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
            "has_refreshes": (run_dir / "refreshes.csv").exists(),
            # Config fields
            "seed": seed,
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


def discover_seeds(
    runs_df: pd.DataFrame,
    pipeline: str,
    variant: str,
) -> List[int]:
    """Return the sorted list of seeds available for a given pipeline/variant."""
    if runs_df.empty:
        return []
    mask = (runs_df["pipeline"] == pipeline) & (runs_df["variant"] == variant)
    sub = runs_df.loc[mask]
    seeds = sub["seed"].dropna().astype(int).unique().tolist()
    return sorted(seeds)


def pick_runs_by_seed(
    runs_df: pd.DataFrame,
    pipeline: str,
    variant: str,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[int, Path]:
    """Return '{seed: latest_run_dir}' for every available (or requested) seed.

    When 'seeds' is 'None' all seeds discovered for the pipeline/variant
    are returned.  Seeds with no matching run are silently dropped.
    """
    out: Dict[int, Path] = {}
    available = set(discover_seeds(runs_df, pipeline, variant))
    wanted = list(seeds) if seeds is not None else sorted(available)
    for s in wanted:
        if s not in available:
            continue
        rd = pick_latest_run(runs_df, pipeline, variant, seed=s)
        if rd is not None:
            out[s] = rd
    return out


# =============================================================================
# Multi-seed aggregation
# =============================================================================


def load_per_seed_csv(
    run_dirs_by_seed: Dict[int, Path],
    csv_name: str,
) -> pd.DataFrame:
    """Concatenate a CSV across runs, tagging each row with its seed.

    Returns an empty DataFrame when no run has the CSV.  The returned frame
    carries a 'seed' column followed by the original CSV columns.
    """
    parts: List[pd.DataFrame] = []
    for seed, rdir in run_dirs_by_seed.items():
        df = read_csv(rdir / csv_name)
        if df is None or df.empty:
            continue
        df = df.copy()
        df.insert(0, "seed", int(seed))
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def aggregate_across_seeds(
    run_dirs_by_seed: Dict[int, Path],
    csv_name: str,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    """Aggregate y_col over seeds at each value of x_col.

    Returns a DataFrame with columns [x_col, mean, std, min, max, n],
    sorted by x_col.  Aggregation is point-wise: seeds that are missing
    a particular x_col value are simply excluded from that point's
    statistics, so n may vary across rows.
    """
    long = load_per_seed_csv(run_dirs_by_seed, csv_name)
    if long.empty or x_col not in long.columns or y_col not in long.columns:
        return pd.DataFrame(columns=[x_col, "mean", "std", "min", "max", "n"])
    grp = long.groupby(x_col)[y_col]
    out = pd.DataFrame({
        "mean": grp.mean(),
        "std": grp.std(ddof=0),
        "min": grp.min(),
        "max": grp.max(),
        "n": grp.count(),
    }).reset_index().sort_values(x_col).reset_index(drop=True)
    return out


def summary_across_seeds(
    run_dirs_by_seed: Dict[int, Path],
) -> pd.DataFrame:
    """Per-seed summary rows using build_summary_row.

    Each seed contributes one row; callers can aggregate further with
    df.agg(['mean', 'std']) etc.
    """
    rows: List[Dict[str, Any]] = []
    for seed in sorted(run_dirs_by_seed):
        row = build_summary_row(label=str(seed), run_dir=run_dirs_by_seed[seed])
        row["seed"] = int(seed)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Filter-family / manifest classification from config
# =============================================================================


def filter_mode(cfg: Dict[str, Any]) -> str:
    """Classify a run into one filter family based on its config.

    Returns one of:
      "none"         -- filter_policy == "none"
      "random"       -- filter_policy == "random"
      "uncertainty"  -- filter_policy == "uncertainty"
      "static"       -- distribution filter with no reference refresh
                        (both window size and reservoir size == 0)
      "window"       -- distribution filter with refresh_window_size > 0
      "reservoir"    -- distribution filter with reservoir_size > 0
      "distribution" -- distribution filter that could not be further
                        classified (config missing the refresh fields)
    """
    fp = str(cfg.get("filter_policy", "") or "").lower()
    if fp in {"", "none"}:
        return "none"
    if fp == "random":
        return "random"
    if fp == "uncertainty":
        return "uncertainty"
    if fp != "distribution":
        return fp
    refresh_every = int(cfg.get("scoring_refresh_every_flushes", 0) or 0)
    refresh_every += int(cfg.get("scoring_refresh_every_rounds", 0) or 0)
    window = int(cfg.get("scoring_refresh_window_size", 0) or 0)
    reservoir = int(cfg.get("scoring_refresh_reservoir_size", 0) or 0)
    if refresh_every <= 0 or (window == 0 and reservoir == 0):
        return "static"
    if reservoir > 0:
        return "reservoir"
    return "window"


# Stable color for each filter family (used across notebooks and the
# package-level CLI: `python -m stream_active_fl.analysis`).
FILTER_FAMILY_COLORS: Dict[str, str] = {
    "none": "#2ca02c",         # green  -- accept-everything upper bound
    "random": "#7f7f7f",       # grey   -- compute-matched null model
    "static": "#1f77b4",       # blue   -- bootstrap-only reference (Mahalanobis)
    "window": "#ff7f0e",       # orange -- sliding window of last M accepts
    "reservoir": "#d62728",    # red    -- uniform reservoir over all accepts
    "uncertainty": "#17becf",  # cyan   -- detection-uncertainty filter
    "distribution": "#9467bd", # purple -- fallback
}


def manifest_family(cfg: Dict[str, Any]) -> str:
    """Return a short tag for the manifest used by a run.

    Inspects 'cfg['manifest_path']' and matches the known filename stems
    (e.g. 'cityday_road_type', 'cityday_curated',
    'citymix_conditions').  Returns '"unknown"' if no pattern matches.
    """
    path = str(cfg.get("manifest_path", "") or "")
    name = os.path.basename(path)
    for tag in (
        "cityday_road_type",
        "cityday_reverse",
        "cityday_curated",
        "citymix_road_type",
        "citymix_reverse",
        "citymix_conditions",
    ):
        if tag in name:
            return tag
    return "unknown"


# =============================================================================
# Bootstrap-size resolution
# =============================================================================


def get_bootstrap_size(
    manifest: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    default: int = 5000,
) -> int:
    """Return the effective bootstrap size for a run.

    Resolution order (most authoritative first):

    1. manifest['ordering']['bootstrap_frames'] -- set by the manifest
       builder for bootstrap-prefixed manifests (e.g.
       manifest_cityday_road_type_boot2000.json).
    2. cfg['bootstrap_frames'] -- the run config value.
    3. default -- used only when neither of the above is available.
    """
    if manifest:
        ordering = manifest.get("ordering") or {}
        n = ordering.get("bootstrap_frames")
        if isinstance(n, int) and n > 0:
            return n
    if cfg:
        n = cfg.get("bootstrap_frames")
        if isinstance(n, int) and n > 0:
            return n
    return default


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


# =============================================================================
# Refresh dynamics (scoring-model refresh events)
# =============================================================================


def load_refreshes(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load refreshes.csv if present.

    Each row logs one scoring-model refresh with the global stream index,
    the number of frames that contributed to the new reference
    (n_reference_frames), and the new threshold.
    """
    return read_csv(run_dir / "refreshes.csv")


def refresh_accept_rate_segments(
    decisions: pd.DataFrame,
    refreshes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Average accept rate inside each inter-refresh segment.

    Returns a DataFrame with columns
    '[segment_start, segment_end, n_frames, n_accepts, accept_rate, refresh_idx]',
    where 'refresh_idx' is the 0-based index of the refresh that *begins*
    the segment ('-1' for the bootstrap-initial segment before the first
    refresh).  This is the primary diagnostic for the "within-block decay"
    behavior of the reservoir filter.
    """
    if decisions.empty or "global_idx" not in decisions.columns:
        return pd.DataFrame(
            columns=["segment_start", "segment_end", "n_frames",
                     "n_accepts", "accept_rate", "refresh_idx"]
        )
    refresh_col = None
    if refreshes is not None and not refreshes.empty:
        for c in ("global_idx", "items_seen"):
            if c in refreshes.columns:
                refresh_col = c
                break
    gi = pd.to_numeric(decisions["global_idx"], errors="coerce")
    gi_min = int(gi.min())
    gi_max = int(gi.max())
    if refresh_col is None:
        n = len(decisions)
        acc = int((decisions["action"] == "accept").sum())
        return pd.DataFrame([{
            "segment_start": gi_min,
            "segment_end": gi_max,
            "n_frames": n,
            "n_accepts": acc,
            "accept_rate": acc / n if n else 0.0,
            "refresh_idx": -1,
        }])
    assert refreshes is not None
    bounds = [gi_min] + [
        int(x) for x in refreshes[refresh_col].tolist()
    ] + [gi_max + 1]
    rows: List[Dict[str, Any]] = []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        sub = decisions.loc[
            (decisions["global_idx"] >= lo) & (decisions["global_idx"] < hi)
        ]
        if sub.empty:
            continue
        n = len(sub)
        acc = int((sub["action"] == "accept").sum())
        rows.append({
            "segment_start": lo,
            "segment_end": hi - 1,
            "n_frames": n,
            "n_accepts": acc,
            "accept_rate": acc / n if n else 0.0,
            "refresh_idx": k - 1,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Per-block accept-rate breakdown
# =============================================================================


def per_block_accept_rate(
    enriched: pd.DataFrame,
    manifest: Optional[Dict[str, Any]],
    bootstrap_frames: int = 0,
) -> pd.DataFrame:
    """Accept rate within each domain block, in block order.

    Uses the manifest's 'ordering.block_order' / 'block_sizes' to slice
    the stream into contiguous segments and computes accept rate, seen
    count, and cumulative accepts per block.

    Returns a DataFrame with columns
    [block_idx, block_label, segment_start, segment_end, n_frames,
    n_accepts, accept_rate, cum_accepts].
    """
    cols = [
        "block_idx", "block_label", "segment_start", "segment_end",
        "n_frames", "n_accepts", "accept_rate", "cum_accepts",
    ]
    if enriched.empty or not manifest:
        return pd.DataFrame(columns=cols)
    trans = block_transitions(manifest, bootstrap_frames=bootstrap_frames)
    if not trans:
        return pd.DataFrame(columns=cols)
    # Extend with sentinel end index
    ordering = manifest.get("ordering") or {}
    block_sizes = ordering.get("block_sizes") or {}
    if block_sizes:
        total_stream = sum(int(v) for v in block_sizes.values())
    elif "global_idx" in enriched.columns:
        total_stream = int(pd.to_numeric(enriched["global_idx"], errors="coerce").max()) + 1
    else:
        total_stream = 0
    bounds: List[Tuple[int, int, str]] = []
    for i, (start, label) in enumerate(trans):
        end = trans[i + 1][0] - 1 if i + 1 < len(trans) else max(start, total_stream - 1)
        bounds.append((start, end, label))
    rows: List[Dict[str, Any]] = []
    cum = 0
    for i, (lo, hi, label) in enumerate(bounds):
        sub = enriched.loc[
            (enriched["global_idx"] >= lo) & (enriched["global_idx"] <= hi)
        ]
        n = len(sub)
        acc = int((sub["action"] == "accept").sum()) if n else 0
        cum += acc
        rows.append({
            "block_idx": i,
            "block_label": label,
            "segment_start": lo,
            "segment_end": hi,
            "n_frames": n,
            "n_accepts": acc,
            "accept_rate": acc / n if n else 0.0,
            "cum_accepts": cum,
        })
    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# Iso-compute mAP comparison
# =============================================================================


def compute_step_series(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """Return a Series of cumulative training-step counts for a run log.

    Works for both streaming ('checkpoints.csv') and federated
    ('rounds.csv') runs.  Resolution order:

    1. 'optimizer_steps' -- cumulative optimizer-step count logged by the
       streaming logger.  Preferred when present.
    2. 'items_processed' -- streaming fallback for older logs that pre-date
       the 'optimizer_steps' column.
    3. sum of 'client_*_optimizer_steps' (per-round, cumsummed) -- federated.

    Returns 'None' when no usable column is available.
    """
    if df is None or df.empty:
        return None
    if "optimizer_steps" in df.columns:
        return pd.to_numeric(df["optimizer_steps"], errors="coerce")
    if "items_processed" in df.columns:
        return pd.to_numeric(df["items_processed"], errors="coerce")
    client_cols = [c for c in df.columns if c.endswith("_optimizer_steps")]
    if client_cols:
        block = df[client_cols].apply(pd.to_numeric, errors="coerce")
        return block.sum(axis=1).cumsum()
    return None


def smoothed_tail_mAP(
    df: Optional[pd.DataFrame],
    k: int = 5,
    mAP_col: str = "mAP",
) -> Optional[float]:
    """Mean of the last 'k' mAP values in a checkpoint log.

    Streaming evaluation is noisy at the small COCO sample sizes used for
    periodic checkpoints (~10k val frames), and a single-checkpoint
    'last_mAP' can swing by +/-0.005.  Averaging the tail flattens this
    noise and gives a more stable end-of-stream estimate suitable for
    leaderboards.

    Returns 'None' when 'df' is empty / missing the column or no usable
    values are present.  The window is capped at the number of available
    rows, so 'k' larger than the trajectory length is silently shrunk.
    """
    if df is None or df.empty or mAP_col not in df.columns:
        return None
    tail = pd.to_numeric(df[mAP_col].tail(int(max(1, k))), errors="coerce").dropna()
    if tail.empty:
        return None
    return float(tail.mean())


def actual_accept_rate(run_dir: Path) -> Optional[float]:
    """Mean accept fraction reported by 'filter_stats.csv'.

    Each filter-stats row is one flush window; 'accept_rate' there is the
    fraction of frames decided in that window that were accepted.  This
    is the *true* streaming accept rate -- the same as 'n_accepts /
    n_decisions' on 'decisions.csv', but available even when the run
    dropped 'decisions.csv' for size.

    Returns 'None' when 'filter_stats.csv' is absent or has no
    'accept_rate' column.
    """
    p = run_dir / "filter_stats.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for col in ("accept_rate", "accepted_rate", "accepts_per_check"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(vals.mean()) if not vals.empty else None
    return None


def iso_compute_mAP(
    ck: Optional[pd.DataFrame],
    target_optim_steps: Optional[int] = None,
    step_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Return mAP-at-target-step and best / last mAP from a run log.

    Works on either streaming checkpoints or federated rounds.  If
    target_optim_steps is given, return the mAP of the last row whose
    effective compute (see compute_step_series) is
    <= target_optim_steps.  When step_col is supplied it takes
    precedence over the auto-detection.

    best_mAP / last_mAP are always populated when the mAP column is
    present; None for ck is accepted as a convenience.
    """
    out: Dict[str, Any] = {"iso_mAP": None, "best_mAP": None, "last_mAP": None}
    if ck is None or ck.empty or "mAP" not in ck.columns:
        return out
    maps = pd.to_numeric(ck["mAP"], errors="coerce")
    if maps.dropna().empty:
        return out
    out["best_mAP"] = float(maps.max())
    out["last_mAP"] = float(maps.dropna().iloc[-1])
    if target_optim_steps is not None:
        if step_col is not None and step_col in ck.columns:
            steps = pd.to_numeric(ck[step_col], errors="coerce")
        else:
            steps = compute_step_series(ck)
        if steps is not None:
            mask = steps <= int(target_optim_steps)
            sub = pd.to_numeric(ck.loc[mask, "mAP"], errors="coerce").dropna()
            if not sub.empty:
                out["iso_mAP"] = float(sub.iloc[-1])
    return out


# =============================================================================
# Aggregated variant summary (filter family aware)
# =============================================================================


def variant_summary_table(
    runs_df: pd.DataFrame,
    pipeline: str,
    variants: Optional[Sequence[str]] = None,
    *,
    target_optim_steps: Union[int, Mapping[str, int], None] = None,
    tail_k: Optional[int] = None,
) -> pd.DataFrame:
    """Build a one-row-per-(variant, seed) summary for a pipeline.

    Columns include the filter family (via filter_mode), the
    manifest family, total accept count / rate, best / last / iso-compute
    mAP, and refresh count.  Designed for the side-by-side variant tables.

    target_optim_steps controls the iso-compute mAP column.  Pass either
    a single int (applied to every variant) or a mapping
    manifest -> steps to use a per-manifest budget (useful when different
    manifests have different no-filter compute totals).
    """
    if variants is None:
        variants = sorted(runs_df.loc[runs_df["pipeline"] == pipeline, "variant"].unique())
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        seed_dirs = pick_runs_by_seed(runs_df, pipeline, variant, seeds=None)
        if not seed_dirs:
            continue
        for seed, rdir in seed_dirs.items():
            cfg = load_run_config(rdir)
            info = load_run_info(rdir)
            ck = read_csv(rdir / "checkpoints.csv")
            rnd = read_csv(rdir / "rounds.csv")
            dec = read_csv(rdir / "decisions.csv")
            ref = load_refreshes(rdir)
            manifest_tag = manifest_family(cfg)
            if isinstance(target_optim_steps, Mapping):
                iso_steps = target_optim_steps.get(manifest_tag)
            else:
                iso_steps = target_optim_steps
            # checkpoints.csv is streaming-only; for federated runs use
            # rounds.csv as the mAP source.
            ck_for_map = ck if (ck is not None and not ck.empty) else rnd
            map_stats = iso_compute_mAP(ck_for_map, target_optim_steps=iso_steps)
            if tail_k is not None and tail_k > 0:
                map_stats["smoothed_mAP"] = smoothed_tail_mAP(ck_for_map, k=tail_k)
            else:
                map_stats["smoothed_mAP"] = None
            if dec is not None and not dec.empty:
                n_dec = int(len(dec))
                n_acc = int((dec["action"] == "accept").sum())
            elif rnd is not None and not rnd.empty:
                # Federated: sum client-level accept / item counts from rounds.csv
                acc_cols = [c for c in rnd.columns if c.endswith("_accepted")]
                item_cols = [c for c in rnd.columns if c.endswith("_items")]
                n_acc = int(rnd[acc_cols].sum().sum()) if acc_cols else 0
                n_dec = int(rnd[item_cols].sum().sum()) if item_cols else 0
            else:
                n_dec, n_acc = 0, 0
            duration_s = info.get("duration_seconds")
            rows.append({
                "variant": variant,
                "seed": int(seed),
                "manifest": manifest_tag,
                "filter_family": filter_mode(cfg),
                "threshold_percentile": cfg.get("threshold_percentile"),
                "refresh_window_size": cfg.get("scoring_refresh_window_size"),
                "reservoir_size": cfg.get("scoring_refresh_reservoir_size"),
                "refresh_every": cfg.get(
                    "scoring_refresh_every_flushes",
                    cfg.get("scoring_refresh_every_rounds"),
                ),
                "n_decisions": n_dec,
                "n_accepts": n_acc,
                "accept_rate": (n_acc / n_dec) if n_dec else 0.0,
                "n_refreshes": 0 if ref is None else int(len(ref)),
                "best_mAP": map_stats["best_mAP"],
                "last_mAP": map_stats["last_mAP"],
                "iso_mAP": map_stats["iso_mAP"],
                "smoothed_mAP": map_stats["smoothed_mAP"],
                "smoothed_tail_k": tail_k,
                "iso_target_steps": iso_steps,
                "actual_accept_rate": actual_accept_rate(rdir),
                "duration_h": (
                    round(float(duration_s) / 3600.0, 2)
                    if isinstance(duration_s, (int, float)) else None
                ),
                "run_dir": str(rdir),
            })
    return pd.DataFrame(rows)


def aggregate_summary_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a variant_summary_table result to mean/std per variant."""
    if df.empty:
        return df
    metric_cols = [c for c in ("accept_rate", "actual_accept_rate",
                                "best_mAP", "last_mAP", "iso_mAP",
                                "smoothed_mAP")
                   if c in df.columns]
    group_cols = ["variant", "manifest", "filter_family",
                  "threshold_percentile", "refresh_window_size",
                  "reservoir_size"]
    group_cols = [c for c in group_cols if c in df.columns]
    agg: Dict[str, Any] = {c: ["mean", "std", "min", "max", "count"]
                           for c in metric_cols}
    if "n_accepts" in df.columns:
        agg["n_accepts"] = ["mean"]
    if "n_refreshes" in df.columns:
        agg["n_refreshes"] = ["mean"]
    out = df.groupby(group_cols, dropna=False).agg(agg)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


# =============================================================================
# Per-domain evaluation loading
# =============================================================================


def load_per_domain_eval(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load long-format per_domain_eval.csv from a retrospective eval run.

    Columns: run_variant, run_seed, run_timestamp, checkpoint,
    checkpoint_mtime, dimension, bucket, n_frames, mAP, mAP_50, mAP_75,
    total_predictions, total_ground_truth, AP_<class>...
    """
    return read_csv(run_dir / "per_domain_eval.csv")


def load_per_domain_checkpoints(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load long-format per_domain_checkpoints.csv from a live streaming run.

    Present only for runs trained with per-domain eval wired into the
    training loop.  Columns: checkpoint_idx, items_processed,
    optimizer_steps, elapsed_seconds, dimension, bucket, n_frames,
    mAP, mAP_50, mAP_75, AP_<class>...
    """
    return read_csv(run_dir / "per_domain_checkpoints.csv")


def aggregate_per_domain_checkpoints(
    run_dirs_by_seed: Mapping[int, Path],
    *,
    target_optim_steps: Optional[int] = None,
    tail_k: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate live 'per_domain_checkpoints.csv' across seeds.

    Picks one row per (seed, dimension, bucket) and averages over seeds.
    Three modes for the per-seed pick:

    1. 'target_optim_steps' is given:
       Take the latest row whose 'optimizer_steps <= target'.
       Useful for iso-compute per-domain comparisons (e.g. report all
       variants at the same step budget so accept-rate differences do
       not confound the comparison).
    2. 'tail_k' is given:
       Average over the last 'k' eval rows for that (dim, bucket) inside
       the seed.  Reduces eval noise; reports end-of-stream behavior.
    3. Neither:
       Take the latest row.  Equivalent to 'tail_k=1'.

    Returns columns 'dimension, bucket, n_frames, mAP_mean, mAP_std,
    mAP_n, optim_steps_mean'.  Empty DataFrame when no rows are found.
    """
    parts: List[pd.DataFrame] = []
    for seed, rdir in run_dirs_by_seed.items():
        df = load_per_domain_checkpoints(rdir)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["seed"] = int(seed)
        if target_optim_steps is not None and "optimizer_steps" in df.columns:
            df = df.loc[
                pd.to_numeric(df["optimizer_steps"], errors="coerce")
                  .le(int(target_optim_steps))
            ].copy()
            if df.empty:
                continue
            picked = (df.sort_values("optimizer_steps")
                        .groupby(["dimension", "bucket"], as_index=False)
                        .tail(1))
            parts.append(picked)
        elif tail_k is not None and tail_k > 0:
            picked = (df.sort_values("optimizer_steps")
                        .groupby(["dimension", "bucket"], as_index=False)
                        .tail(int(tail_k))
                        .groupby(["dimension", "bucket"], as_index=False)
                        .agg(mAP=("mAP", "mean"),
                             n_frames=("n_frames", "max"),
                             optimizer_steps=("optimizer_steps", "mean")))
            picked["seed"] = int(seed)
            parts.append(picked)
        else:
            picked = (df.sort_values("optimizer_steps")
                        .groupby(["dimension", "bucket"], as_index=False)
                        .tail(1))
            parts.append(picked)
    if not parts:
        return pd.DataFrame(
            columns=["dimension", "bucket", "n_frames", "mAP_mean",
                     "mAP_std", "mAP_n", "optim_steps_mean"]
        )
    long = pd.concat(parts, ignore_index=True)
    grp = long.groupby(["dimension", "bucket"])["mAP"]
    out = pd.DataFrame({
        "mAP_mean": grp.mean(),
        "mAP_std": grp.std(ddof=0),
        "mAP_n": grp.count(),
    }).reset_index()
    n_frames = (long.groupby(["dimension", "bucket"])["n_frames"].max()
                    .reset_index().rename(columns={"n_frames": "n_frames"}))
    steps_col = "optimizer_steps"
    if steps_col in long.columns:
        steps = (long.groupby(["dimension", "bucket"])[steps_col].mean()
                     .reset_index()
                     .rename(columns={steps_col: "optim_steps_mean"}))
        out = out.merge(steps, on=["dimension", "bucket"], how="left")
    out = out.merge(n_frames, on=["dimension", "bucket"], how="left")
    return out


def per_domain_checkpoints_table(
    runs_df: pd.DataFrame,
    pipeline: str,
    variants: Sequence[str],
    *,
    target_optim_steps: Optional[int] = None,
    tail_k: Optional[int] = None,
) -> pd.DataFrame:
    """Per-domain leaderboard across variants from live checkpoint logs.

    For each variant in 'variants', aggregate 'per_domain_checkpoints.csv'
    across its seeds (see 'aggregate_per_domain_checkpoints' for the
    'target_optim_steps' / 'tail_k' modes), then concatenate with a
    'run_variant' column so the result is a long-format table suitable
    for downstream merges with 'per_domain_gain_vs_baseline' and
    'balanced_map_table'.
    """
    parts: List[pd.DataFrame] = []
    for v in variants:
        seed_dirs = pick_runs_by_seed(runs_df, pipeline, v, seeds=None)
        if not seed_dirs:
            continue
        df = aggregate_per_domain_checkpoints(
            seed_dirs,
            target_optim_steps=target_optim_steps,
            tail_k=tail_k,
        )
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "run_variant", v)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def iso_accept_pairs(
    summary_df: pd.DataFrame,
    *,
    accept_col: str = "accept_rate",
    family_col: str = "filter_family",
    tolerance: float = 0.03,
    reference_family: str = "random",
) -> pd.DataFrame:
    """Match each filter run to the reference run with the closest accept rate.

    Use this to build "iso-accept" comparisons: for every non-reference
    variant, the row reports the variant's accept rate, the reference
    variant whose mean accept rate is closest, and the absolute accept-
    rate gap.  Variants without a reference match within 'tolerance'
    (e.g. filters that accept 33% with no random_p33 baseline available)
    get 'matched_variant=None' so callers can flag them as
    "no fair comparison".

    The intended caller is something like:

        smry = ah.aggregate_summary_across_seeds(
            ah.variant_summary_table(runs_df, "streaming", variants))
        pairs = ah.iso_accept_pairs(smry)

    Returns columns
    'variant, filter_family, accept_rate, matched_variant,
    matched_accept_rate, gap'.
    """
    if summary_df.empty:
        return pd.DataFrame()
    accept_col_eff = accept_col if accept_col in summary_df.columns else f"{accept_col}_mean"
    if accept_col_eff not in summary_df.columns or family_col not in summary_df.columns:
        return pd.DataFrame()
    refs = summary_df.loc[summary_df[family_col] == reference_family,
                          ["variant", accept_col_eff]].copy()
    refs = refs.rename(columns={accept_col_eff: "ref_accept"})
    rows: List[Dict[str, Any]] = []
    for _, r in summary_df.iterrows():
        if r[family_col] == reference_family:
            continue
        a = float(r[accept_col_eff])
        if refs.empty or pd.isna(a):
            rows.append({
                "variant": r["variant"],
                "filter_family": r[family_col],
                "accept_rate": a,
                "matched_variant": None,
                "matched_accept_rate": None,
                "gap": None,
            })
            continue
        gaps = (refs["ref_accept"] - a).abs()
        i = int(gaps.idxmin())
        gap = float(gaps.loc[i])
        rows.append({
            "variant": r["variant"],
            "filter_family": r[family_col],
            "accept_rate": a,
            "matched_variant": refs.loc[i, "variant"] if gap <= tolerance else None,
            "matched_accept_rate": float(refs.loc[i, "ref_accept"])
                                    if gap <= tolerance else None,
            "gap": gap,
        })
    return pd.DataFrame(rows)


def collect_per_domain_eval(run_dirs: Iterable[Path]) -> pd.DataFrame:
    """Concatenate per_domain_eval.csv across many run dirs, skipping misses."""
    parts: List[pd.DataFrame] = []
    for rdir in run_dirs:
        df = load_per_domain_eval(rdir)
        if df is None or df.empty:
            continue
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _extract_variant_seed(run_dir: Path) -> Tuple[str, Optional[int]]:
    """Infer variant dir name and seed integer from an experiment run path."""
    try:
        seed = int(run_dir.parent.name.replace("seed_", ""))
    except ValueError:
        seed = None
    variant = run_dir.parent.parent.name
    return variant, seed


def aggregate_per_domain_eval(
    run_dirs: Iterable[Path],
    checkpoint: str = "final",
) -> pd.DataFrame:
    """Aggregate per-(variant, dimension, bucket) mAP over seeds.

    Reads per_domain_eval.csv from each run dir, filters to the given
    checkpoint ('best' or 'final'), and returns mean / std / n across seeds.

    Output columns: run_variant, dimension, bucket, n_frames,
    mAP_mean, mAP_std, mAP_n, mAP_50_mean, mAP_75_mean, plus per-class means.
    """
    parts: List[pd.DataFrame] = []
    for rdir in run_dirs:
        df = load_per_domain_eval(rdir)
        if df is None or df.empty:
            continue
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.loc[df["checkpoint"] == checkpoint].copy()
    if df.empty:
        return df

    metric_cols = [c for c in df.columns if c.startswith(("mAP", "AP_"))]
    group = ["run_variant", "dimension", "bucket"]
    agg: Dict[str, Any] = {c: ["mean", "std", "count"] for c in metric_cols}
    agg["n_frames"] = ["max"]
    out = df.groupby(group, dropna=False).agg(agg)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.rename(columns={"n_frames_max": "n_frames"})
    return out.reset_index()


def per_domain_gain_vs_baseline(
    per_domain: pd.DataFrame,
    baseline_variant: str,
    metric: str = "mAP_mean",
) -> pd.DataFrame:
    """For each (variant, dimension, bucket), subtract the baseline's value.

    Produces a variant-vs-baseline delta table: positive 'gain' means the
    variant outperforms the baseline on that bucket.  Useful for
    night_gain / snow_gain / rural_gain / day_cost headlines.
    """
    if per_domain.empty or metric not in per_domain.columns:
        return pd.DataFrame()
    base = (
        per_domain.loc[per_domain["run_variant"] == baseline_variant,
                       ["dimension", "bucket", metric]]
        .rename(columns={metric: "baseline"})
    )
    merged = per_domain.merge(base, on=["dimension", "bucket"], how="left")
    merged["gain"] = merged[metric] - merged["baseline"]
    return merged


def balanced_map_table(
    per_domain: pd.DataFrame,
    metric: str = "mAP_mean",
) -> pd.DataFrame:
    """Unweighted mean of per-bucket mAPs within each dimension, per variant.

    Returned long format: run_variant, dimension, balanced_mAP, worst_mAP,
    n_buckets.  balanced_mAP is the average across reported buckets
    (which excludes any bucket below min_bucket_size); worst_mAP is the
    minimum across reported buckets (tail-performance proxy).
    """
    if per_domain.empty or metric not in per_domain.columns:
        return pd.DataFrame()
    df = per_domain.loc[per_domain["dimension"] != "aggregate"].copy()
    grp = df.groupby(["run_variant", "dimension"])[metric]
    out = pd.DataFrame({
        "balanced_mAP": grp.mean(),
        "worst_mAP": grp.min(),
        "n_buckets": grp.count(),
    }).reset_index()
    return out


# =============================================================================
# Shared visualization constants
# =============================================================================

# Canonical colors for road_type blocks.  Used by notebooks 00 and 02.
DOMAIN_COLORS: Dict[str, str] = {
    "city": "#1f77b4",
    "arterial-urban": "#ff7f0e",
    "highway": "#2ca02c",
    "arterial-rural": "#d62728",
    "smaller-rural": "#9467bd",
}

# Canonical colors for time_of_day values.
TOD_COLORS: Dict[str, str] = {
    "day": "#FFD700",
    "twilight": "#FF8C00",
    "dawn/dusk": "#FF8C00",
    "night": "#191970",
}

# Canonical colors for weather / conditions blocks.
WEATHER_COLORS: Dict[str, str] = {
    "clear": "#FDDA0D",
    "cloudy": "#A9A9A9",
    "partly_cloudy": "#BDC3C7",
    "fog": "#C4C3D0",
    "rain_wet": "#4682B4",
    "rain": "#4682B4",
    "snow": "#E0F0FF",
    "wet": "#5DADE2",
    "dry": "#F5CBA7",
    "unknown": "#CCCCCC",
}

# Short display names for road_type categories.
ROAD_SHORT: Dict[str, str] = {
    "city": "City",
    "arterial-urban": "Art.-Urban",
    "highway": "Highway",
    "arterial-rural": "Art.-Rural",
    "smaller-rural": "Sm.-Rural",
}

# Short display names for weather categories.
WEATHER_SHORT: Dict[str, str] = {
    "clear": "Clear",
    "cloudy": "Cloudy",
    "fog": "Fog",
    "rain_wet": "Rain/Wet",
    "snow": "Snow",
}

# Short display names for time_of_day categories.
TOD_SHORT: Dict[str, str] = {
    "day": "Day",
    "twilight": "Twilight",
    "dawn/dusk": "Dawn/Dusk",
    "night": "Night",
}


def color_map_for_field(field: str) -> Dict[str, str]:
    """Return the canonical color map for a metadata field name."""
    if field == "road_type":
        return DOMAIN_COLORS
    if field == "time_of_day":
        return TOD_COLORS
    if field in ("scraped_weather", "weather"):
        return WEATHER_COLORS
    return {}


def short_name(value: str) -> str:
    """Look up a short display name across road / weather / time-of-day maps."""
    return ROAD_SHORT.get(value, WEATHER_SHORT.get(value, TOD_SHORT.get(value, value)))


# =============================================================================
# Bootstrap / stream split
# =============================================================================


def bootstrap_train_frames(manifest: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """Return the first n train frames from a manifest (skipping any val frames).

    This matches what the experiment code actually uses for the bootstrap
    prefix; slicing the raw 'frames' list by position is incorrect because
    val frames are interleaved.
    """
    out: List[Dict[str, Any]] = []
    for f in manifest.get("frames", []):
        if f.get("split") == "train":
            out.append(f)
            if len(out) >= n:
                break
    return out


def split_bootstrap_stream(
    manifest: Dict[str, Any], bootstrap_frames: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (bootstrap_df, stream_df) of train frames from a manifest."""
    mdf = manifest_to_dataframe(manifest)
    train = mdf.loc[mdf["split"] == "train"].reset_index(drop=True)
    return (
        train.iloc[:bootstrap_frames].reset_index(drop=True),
        train.iloc[bootstrap_frames:].reset_index(drop=True),
    )


# =============================================================================
# Federated client partitioning
# =============================================================================


def primary_block_field(manifest: Dict[str, Any]) -> str:
    """Pick the frame-metadata field that matches the manifest's block field."""
    strategy = str(manifest.get("ordering", {}).get("strategy", ""))
    if "conditions" in strategy:
        return "scraped_weather"
    return "road_type"


def client_ranges_from_config(
    manifest: Dict[str, Any],
    cfg: Dict[str, Any],
    bootstrap_frames: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], List[List[str]], str]:
    """Reconstruct the per-client stream ranges a federated run used.

    Reads partition_strategy and (for "domain_aligned")
    domain_client_groups from the run config.  When bootstrap_frames
    is None it is resolved from the manifest / config via
    get_bootstrap_size.

    Returns:
        (ranges, groups, strategy)
            ranges: list of (start, end) in post-bootstrap stream indices
            groups: list of block-name lists per client (empty for contiguous)
            strategy: the partition_strategy string from the config
    """
    if bootstrap_frames is None:
        bootstrap_frames = get_bootstrap_size(manifest, cfg)
    ordering = manifest.get("ordering", {})
    block_order = list(ordering.get("block_order", []))
    block_sizes = ordering.get("block_sizes", {})
    n_train = sum(1 for f in manifest.get("frames", []) if f.get("split") == "train")
    n_stream = max(0, n_train - bootstrap_frames)
    n_clients = int(cfg.get("num_clients", 4))
    strategy = str(cfg.get("partition_strategy", "contiguous"))

    if strategy == "domain_aligned":
        groups = cfg.get("domain_client_groups") or []
        if len(groups) != n_clients:
            raise ValueError(
                f"domain_client_groups has {len(groups)} entries but "
                f"num_clients={n_clients}"
            )
        block_offset: Dict[str, int] = {}
        pos = 0
        for b in block_order:
            block_offset[b] = pos
            pos += int(block_sizes.get(b, 0))
        ranges: List[Tuple[int, int]] = []
        out_groups: List[List[str]] = []
        for group in groups:
            starts = [block_offset[b] for b in group]
            ends = [block_offset[b] + int(block_sizes[b]) for b in group]
            ranges.append((min(starts), max(ends)))
            out_groups.append(list(group))
        return ranges, out_groups, strategy

    # Contiguous fallback
    ranges_c: List[Tuple[int, int]] = []
    start = 0
    for cid in range(n_clients):
        base = n_stream // n_clients
        extra = 1 if cid < (n_stream % n_clients) else 0
        end = start + base + extra
        ranges_c.append((start, end))
        start = end
    return ranges_c, [[] for _ in range(n_clients)], strategy


def client_partitions(
    manifest: Dict[str, Any],
    cfg: Dict[str, Any],
    bootstrap_frames: Optional[int] = None,
) -> Dict[int, pd.DataFrame]:
    """Return per-client stream-frame DataFrames using the run config's strategy.

    bootstrap_frames defaults to get_bootstrap_size.
    """
    if bootstrap_frames is None:
        bootstrap_frames = get_bootstrap_size(manifest, cfg)
    _, stream = split_bootstrap_stream(manifest, bootstrap_frames)
    ranges, _, _ = client_ranges_from_config(manifest, cfg, bootstrap_frames)
    return {
        cid: stream.iloc[s:e].reset_index(drop=True)
        for cid, (s, e) in enumerate(ranges)
    }


def client_domain_labels(
    manifest: Dict[str, Any],
    cfg: Dict[str, Any],
    bootstrap_frames: Optional[int] = None,
    field: Optional[str] = None,
    top_n: int = 2,
) -> Dict[int, str]:
    """Build domain-annotated labels like 'Client 0 (city 100%)'.

    If 'field' is None it is picked from the manifest's block strategy.
    """
    if field is None:
        field = primary_block_field(manifest)
    parts = client_partitions(manifest, cfg, bootstrap_frames)
    labels: Dict[int, str] = {}
    for cid in sorted(parts):
        df = parts[cid]
        if field not in df.columns or df.empty:
            labels[cid] = f"Client {cid}"
            continue
        counts = Counter(df[field].dropna().astype(str))
        total = sum(counts.values())
        if total == 0:
            labels[cid] = f"Client {cid}"
            continue
        top = counts.most_common(top_n)
        desc = ", ".join(f"{c} {n/total:.0%}" for c, n in top)
        labels[cid] = f"Client {cid} ({desc})"
    return labels


# =============================================================================
# Notebook style
# =============================================================================


def setup_notebook_style(dpi: int = 140, base_font_size: int = 9) -> None:
    """Apply consistent matplotlib rcParams across analysis notebooks."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": dpi,
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size,
        "legend.fontsize": base_font_size - 1,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "figure.facecolor": "white",
    })
