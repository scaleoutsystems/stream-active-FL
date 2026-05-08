"""
Streaming-specific analysis tables.

Consolidates the headline tables, per-domain grids, iso-accept pairings,
and ablation comparisons used by the streaming notebook
(`notebooks/01_streaming_analysis.ipynb`) and by
`python -m stream_active_fl.analysis`.  All helpers operate on the
standard layout

    outputs/streaming/<variant>/seed_<N>/<YYYY-mm-dd_HH-MM-SS>/

via the run-discovery and CSV-loading primitives in `runs`.

Public surface (the things the notebook actually calls):

    Registry / lookups
        FEATURED_VARIANTS              -- ordered list of canonical variants
        VARIANT_LABEL                  -- variant -> short display label
        VARIANT_FAMILY                 -- variant -> filter family
        ISO_ACCEPT_PAIRINGS            -- explicit filter <-> random pairs
        ABLATION_PAIRINGS              -- single <-> ablation pairs
        latest_seed_dir(variant, seed) -- newest run dir for a (variant, seed)
        variant_seed_dirs(variant)     -- {seed: run_dir} across all seeds
        manifest_for_variant(v)        -- "curated" / "temporal" / ...

    Headline tables
        inventory_table(variants)
        ablation_pair_table(pairings)
        iso_accept_table(pairings)

    Per-domain tables
        per_domain_grid(variants, dim, k)
        per_domain_summary(variants, dim, k)
        per_domain_delta_grid(pairings, dim, k)
        per_block_routing(variants, project_root)

    Trajectory data
        per_domain_trajectory(variant, blocks, dim)
        mAP_trajectory(variants)
        windowed_accept_rate_aggregated(variant, window)
        stream_composition(manifest, fields, window)

    All numeric helpers return tidy long-format pandas DataFrames so they
    drop straight into figure-making code without further reshaping.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

import numpy as np
import pandas as pd

from . import runs as ah


# =============================================================================
# Variant registry
# =============================================================================

# Ordered list of variants featured in the streaming write-up.  The order
# is also the row order for tables and the legend order for figures.
FEATURED_VARIANTS: List[str] = [
    # Reference baselines (curated)
    "no_filter_cityday_curated",
    "random_p17_cityday_curated",
    "random_p21_cityday_curated",
    "random_p26_cityday_curated",
    "random_p29_cityday_curated",
    "random_p33_cityday_curated",
    "random_p73_cityday_curated",
    "random_p77_cityday_curated",
    # Static distribution filter
    "static_p15_cityday_curated",
    "static_p20_cityday_curated",
    # Adaptive Window (curated)
    "adaptive_window_p15_cityday_curated",
    "adaptive_window_p20_cityday_curated",
    "adaptive_window_p20_twoRef_cityday_curated",
    "adaptive_window_p20_noBoot_cityday_curated",
    # Adaptive Reservoir (curated)
    "adaptive_reservoir_p15_cityday_curated",
    "adaptive_reservoir_p20_cityday_curated",
    "adaptive_reservoir_p20_twoRef_cityday_curated",
    "adaptive_reservoir_p20_noBoot_cityday_curated",
    # Matched-memory window-vs-reservoir comparison at memory size 1500
    # (mirrors the federated chapter's per-client memory budget).
    "adaptive_window_p20_m1500_cityday_curated",
    "adaptive_window_p20_twoRef_m1500_cityday_curated",
    "adaptive_reservoir_p20_m1500_cityday_curated",
    "adaptive_reservoir_p20_twoRef_m1500_cityday_curated",
    # Cross-stream-order replication (temporal)
    "no_filter_cityday_temporal",
    "random_p21_cityday_temporal",
    "random_p28_cityday_temporal",
    "random_p31_cityday_temporal",
    "adaptive_window_p20_cityday_temporal",
    "adaptive_window_p20_twoRef_cityday_temporal",
    "adaptive_reservoir_p20_cityday_temporal",
    "adaptive_reservoir_p20_twoRef_cityday_temporal",
]

# Short display labels keyed by variant name.  Missing entries fall back
# to the variant name itself (see `label_for`).
VARIANT_LABEL: Dict[str, str] = {
    # curated reference
    "no_filter_cityday_curated":                          "none",
    "random_p17_cityday_curated":                         "random p17",
    "random_p21_cityday_curated":                         "random p21",
    "random_p26_cityday_curated":                         "random p26",
    "random_p29_cityday_curated":                         "random p29",
    "random_p33_cityday_curated":                         "random p33",
    "random_p73_cityday_curated":                         "random p73",
    "random_p77_cityday_curated":                         "random p77",
    "static_p15_cityday_curated":                         "static p15",
    "static_p20_cityday_curated":                         "static p20",
    # adaptive window
    "adaptive_window_p15_cityday_curated":                "window p15",
    "adaptive_window_p20_cityday_curated":                "window p20",
    "adaptive_window_p20_twoRef_cityday_curated":         "window p20 twoRef",
    "adaptive_window_p20_noBoot_cityday_curated":         "window p20 noBoot",
    # adaptive reservoir
    "adaptive_reservoir_p15_cityday_curated":             "reservoir p15",
    "adaptive_reservoir_p20_cityday_curated":             "reservoir p20",
    "adaptive_reservoir_p20_twoRef_cityday_curated":      "reservoir p20 twoRef",
    "adaptive_reservoir_p20_noBoot_cityday_curated":      "reservoir p20 noBoot",
    # matched-memory (m1500)
    "adaptive_window_p20_m1500_cityday_curated":          "window p20 (m1500)",
    "adaptive_window_p20_twoRef_m1500_cityday_curated":   "window p20 twoRef (m1500)",
    "adaptive_reservoir_p20_m1500_cityday_curated":       "reservoir p20 (m1500)",
    "adaptive_reservoir_p20_twoRef_m1500_cityday_curated":
                                                          "reservoir p20 twoRef (m1500)",
    # temporal
    "no_filter_cityday_temporal":                         "none (T)",
    "random_p21_cityday_temporal":                        "random p21 (T)",
    "random_p28_cityday_temporal":                        "random p28 (T)",
    "random_p31_cityday_temporal":                        "random p31 (T)",
    "adaptive_window_p20_cityday_temporal":               "window p20 (T)",
    "adaptive_window_p20_twoRef_cityday_temporal":        "window p20 twoRef (T)",
    "adaptive_reservoir_p20_cityday_temporal":            "reservoir p20 (T)",
    "adaptive_reservoir_p20_twoRef_cityday_temporal":     "reservoir p20 twoRef (T)",
}


# Print-friendly variant labels (used by ``label_for``).  Memory-size
# suffixes (``m1500``) are stripped because the body text introduces the
# per-client memory budget once; the ``twoRef`` policy is spelled out
# as ``two-ref``; ``(T)`` is expanded to "(temporal)" so casual readers
# can decode it without consulting the legend.
THESIS_LABEL: Dict[str, str] = {
    "no_filter_cityday_curated":                          "No filter",
    "random_p17_cityday_curated":                         r"Random ($\rho{=}0.17$)",
    "random_p21_cityday_curated":                         r"Random ($\rho{=}0.21$)",
    "random_p23_cityday_curated":                         r"Random ($\rho{=}0.23$)",
    "random_p26_cityday_curated":                         r"Random ($\rho{=}0.26$)",
    "random_p27_cityday_curated":                         r"Random ($\rho{=}0.27$)",
    "random_p29_cityday_curated":                         r"Random ($\rho{=}0.29$)",
    "random_p33_cityday_curated":                         r"Random ($\rho{=}0.33$)",
    "random_p73_cityday_curated":                         r"Random ($\rho{=}0.73$)",
    "random_p77_cityday_curated":                         r"Random ($\rho{=}0.77$)",
    "static_p15_cityday_curated":                         r"Static ($\tau_{15}$)",
    "static_p20_cityday_curated":                         r"Static ($\tau_{20}$)",
    "adaptive_window_p15_cityday_curated":                "Window single-ref ($\\tau_{15}$)",
    "adaptive_window_p20_cityday_curated":                "Window single-ref",
    "adaptive_window_p20_twoRef_cityday_curated":         "Window two-ref",
    "adaptive_window_p20_noBoot_cityday_curated":         "Window (no anchor)",
    "adaptive_reservoir_p15_cityday_curated":             "Reservoir single-ref ($\\tau_{15}$)",
    "adaptive_reservoir_p20_cityday_curated":             "Reservoir single-ref",
    "adaptive_reservoir_p20_twoRef_cityday_curated":      "Reservoir two-ref",
    "adaptive_reservoir_p20_noBoot_cityday_curated":      "Reservoir (no anchor)",
    "adaptive_window_p20_m1500_cityday_curated":          "Window single-ref",
    "adaptive_window_p20_twoRef_m1500_cityday_curated":   "Window two-ref",
    "adaptive_reservoir_p20_m1500_cityday_curated":       "Reservoir single-ref",
    "adaptive_reservoir_p20_twoRef_m1500_cityday_curated":"Reservoir two-ref",
    "no_filter_cityday_temporal":                         "No filter (temporal)",
    "random_p21_cityday_temporal":                        r"Random ($\rho{=}0.21$, temporal)",
    "random_p28_cityday_temporal":                        r"Random ($\rho{=}0.28$, temporal)",
    "random_p31_cityday_temporal":                        r"Random ($\rho{=}0.31$, temporal)",
    "adaptive_window_p20_cityday_temporal":               "Window single-ref (temporal)",
    "adaptive_window_p20_twoRef_cityday_temporal":        "Window two-ref (temporal)",
    "adaptive_reservoir_p20_cityday_temporal":            "Reservoir single-ref (temporal)",
    "adaptive_reservoir_p20_twoRef_cityday_temporal":     "Reservoir two-ref (temporal)",
}


def label_for(variant: str) -> str:
    """Canonical print-friendly label for a variant (used in tables and figures).

    Looks up `THESIS_LABEL` first, falls back to the original
    notebook-style entry in `VARIANT_LABEL`, and finally to the variant
    string itself.  Used everywhere -- table column headers, plot
    legends, scatter annotations.
    """
    return THESIS_LABEL.get(variant, VARIANT_LABEL.get(variant, variant))


def manifest_for_variant(variant: str) -> str:
    """Return ``"curated"``, ``"temporal"``, or ``"unknown"`` from the name."""
    if "cityday_curated" in variant:
        return "curated"
    if "cityday_temporal" in variant:
        return "temporal"
    return "unknown"


def family_for_variant(variant: str, project_root: Optional[Path] = None) -> str:
    """Filter family from the run config; ``"unknown"`` when not discoverable."""
    rdir = latest_seed_dir(variant, seed=None, project_root=project_root)
    if rdir is None:
        # Heuristic from name when no run dir is available.
        if variant.startswith("no_filter"):
            return "none"
        if variant.startswith("random_"):
            return "random"
        if variant.startswith("static_"):
            return "static"
        if "noBoot" in variant or "twoRef" in variant or "_window_" in variant:
            return "window" if "_window_" in variant else (
                "reservoir" if "_reservoir_" in variant else "distribution")
        if "_reservoir_" in variant:
            return "reservoir"
        return "unknown"
    return ah.filter_mode(ah.load_run_config(rdir))


# Per-variant filter family (loaded lazily on first access).
VARIANT_FAMILY: Dict[str, str] = {}  # populated by prime_registry().


# Iso-accept pairings used in the streaming write-up.  Each tuple is
# (filter_variant, random_variant) where the random partner's
# ``accept_fraction`` is sized to match the filter's empirical accept
# rate (gap typically < 0.03).
ISO_ACCEPT_PAIRINGS: List[Tuple[str, str]] = [
    ("static_p15_cityday_curated",                       "random_p73_cityday_curated"),
    ("static_p20_cityday_curated",                       "random_p77_cityday_curated"),
    ("adaptive_window_p15_cityday_curated",              "random_p29_cityday_curated"),
    ("adaptive_window_p20_cityday_curated",              "random_p33_cityday_curated"),
    ("adaptive_window_p20_twoRef_cityday_curated",       "random_p33_cityday_curated"),
    ("adaptive_window_p20_noBoot_cityday_curated",       "random_p33_cityday_curated"),
    ("adaptive_reservoir_p15_cityday_curated",           "random_p17_cityday_curated"),
    ("adaptive_reservoir_p20_cityday_curated",           "random_p21_cityday_curated"),
    ("adaptive_reservoir_p20_twoRef_cityday_curated",    "random_p21_cityday_curated"),
    ("adaptive_reservoir_p20_noBoot_cityday_curated",    "random_p17_cityday_curated"),
    # Matched-memory variants (m1500) - empirical accept rates landed at
    # window 0.260 / window twoRef 0.270 / reservoir 0.233 / reservoir twoRef
    # 0.210, so we pair against the closest existing random.  random_p26 was
    # added (3 seeds) to give the window m1500 single-ref filter a tighter
    # iso-accept partner than random_p29 (gap 0.029 -> 0.000).
    ("adaptive_window_p20_m1500_cityday_curated",        "random_p26_cityday_curated"),
    ("adaptive_window_p20_twoRef_m1500_cityday_curated", "random_p29_cityday_curated"),
    ("adaptive_reservoir_p20_m1500_cityday_curated",     "random_p21_cityday_curated"),
    ("adaptive_reservoir_p20_twoRef_m1500_cityday_curated", "random_p21_cityday_curated"),
    ("adaptive_window_p20_cityday_temporal",             "random_p28_cityday_temporal"),
    ("adaptive_window_p20_twoRef_cityday_temporal",      "random_p31_cityday_temporal"),
    ("adaptive_reservoir_p20_cityday_temporal",          "random_p21_cityday_temporal"),
    ("adaptive_reservoir_p20_twoRef_cityday_temporal",   "random_p21_cityday_temporal"),
]


# Ablation pairings: ``(label, baseline_variant, ablated_variant)``.
ABLATION_PAIRINGS: Dict[str, List[Tuple[str, str, str]]] = {
    "twoRef": [
        ("Win_p20  (curated)",
         "adaptive_window_p20_cityday_curated",
         "adaptive_window_p20_twoRef_cityday_curated"),
        ("Res_p20  (curated)",
         "adaptive_reservoir_p20_cityday_curated",
         "adaptive_reservoir_p20_twoRef_cityday_curated"),
        ("Win_p20  (temporal)",
         "adaptive_window_p20_cityday_temporal",
         "adaptive_window_p20_twoRef_cityday_temporal"),
        ("Res_p20  (temporal)",
         "adaptive_reservoir_p20_cityday_temporal",
         "adaptive_reservoir_p20_twoRef_cityday_temporal"),
    ],
    "noBoot": [
        ("Win_p20",
         "adaptive_window_p20_cityday_curated",
         "adaptive_window_p20_noBoot_cityday_curated"),
        ("Res_p20",
         "adaptive_reservoir_p20_cityday_curated",
         "adaptive_reservoir_p20_noBoot_cityday_curated"),
    ],
    "static_vs_adaptive": [
        ("p20",
         "static_p20_cityday_curated",
         "adaptive_window_p20_twoRef_cityday_curated"),
    ],
    # Window vs reservoir at MATCHED memory budget (1500), to decouple
    # the algorithm from the memory size.  Streaming defaults
    # historically used window=1000 / reservoir=2000.
    "matched_memory": [
        ("Win vs Res  p20  (m1500)",
         "adaptive_window_p20_m1500_cityday_curated",
         "adaptive_reservoir_p20_m1500_cityday_curated"),
        ("Win vs Res  p20 twoRef  (m1500)",
         "adaptive_window_p20_twoRef_m1500_cityday_curated",
         "adaptive_reservoir_p20_twoRef_m1500_cityday_curated"),
    ],
}


# =============================================================================
# Run-dir resolution
# =============================================================================

def latest_seed_dir(
    variant: str,
    seed: Optional[int] = None,
    *,
    project_root: Optional[Path] = None,
) -> Optional[Path]:
    """Return the most-recent run directory for a (variant, seed).

    When ``seed`` is ``None`` the lowest-numbered seed available is used,
    so callers that just need *any* run dir get a deterministic answer.
    """
    root = (project_root or ah.find_project_root()) / "outputs" / "streaming" / variant
    if not root.is_dir():
        return None
    if seed is None:
        seed_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")]
        if not seed_dirs:
            return None
        seed_dirs.sort(key=lambda p: int(p.name.split("_", 1)[1]))
        base = seed_dirs[0]
    else:
        base = root / f"seed_{seed}"
        if not base.is_dir():
            return None
    cands = sorted(p for p in base.iterdir() if p.is_dir())
    return cands[-1] if cands else None


def variant_seed_dirs(
    variant: str,
    *,
    seeds: Sequence[int] = (42, 43, 44),
    project_root: Optional[Path] = None,
) -> Dict[int, Path]:
    """``{seed: latest_run_dir}`` over the requested seeds (missing seeds skipped)."""
    out: Dict[int, Path] = {}
    for s in seeds:
        d = latest_seed_dir(variant, s, project_root=project_root)
        if d is not None:
            out[s] = d
    return out


def prime_registry(project_root: Optional[Path] = None) -> None:
    """Populate `VARIANT_FAMILY` for every variant in `FEATURED_VARIANTS`."""
    for v in FEATURED_VARIANTS:
        VARIANT_FAMILY[v] = family_for_variant(v, project_root=project_root)


# =============================================================================
# Per-variant statistics (single number per variant, averaged over seeds)
# =============================================================================

def _stats(
    variant: str,
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
    tail_k: int = 5,
) -> Dict[str, float]:
    """Compute headline statistics for a variant (mean across seeds).

    Returns dict with keys ``n, accept, accept_std, smoothed, smoothed_std,
    final, steps``.  ``smoothed`` averages the last ``tail_k`` overall mAP
    checkpoints.  ``steps`` is the cumulative optimizer-step count at the
    final checkpoint, averaged over seeds.
    """
    sd = variant_seed_dirs(variant, seeds=seeds, project_root=project_root)
    accs, smoothed, finals, steps = [], [], [], []
    for run in sd.values():
        ar = ah.actual_accept_rate(run)
        if ar is not None:
            accs.append(ar)
        ck = ah.read_csv(run / "checkpoints.csv")
        if ck is not None and not ck.empty:
            finals.append(float(ck["mAP"].iloc[-1]))
            s = ah.smoothed_tail_mAP(ck, k=tail_k)
            if s is not None:
                smoothed.append(s)
            if "optimizer_steps" in ck.columns:
                steps.append(int(ck["optimizer_steps"].iloc[-1]))
    return {
        "n": len(sd),
        "accept": float(np.mean(accs)) if accs else float("nan"),
        "accept_std": float(np.std(accs)) if len(accs) > 1 else float("nan"),
        "smoothed": float(np.mean(smoothed)) if smoothed else float("nan"),
        "smoothed_std": float(np.std(smoothed)) if len(smoothed) > 1 else float("nan"),
        "final": float(np.mean(finals)) if finals else float("nan"),
        "steps": int(np.mean(steps)) if steps else None,
    }


# =============================================================================
# Headline tables
# =============================================================================

def inventory_table(
    variants: Optional[Sequence[str]] = None,
    *,
    project_root: Optional[Path] = None,
    tail_k: int = 5,
) -> pd.DataFrame:
    """One row per variant: accept rate, smoothed tail-k mAP, optim steps."""
    variants = list(variants) if variants is not None else list(FEATURED_VARIANTS)
    rows: List[Dict] = []
    for v in variants:
        s = _stats(v, project_root=project_root, tail_k=tail_k)
        if s["n"] == 0:
            continue
        rows.append({
            "variant": v,
            "label": label_for(v),
            "manifest": manifest_for_variant(v),
            "family": family_for_variant(v, project_root=project_root),
            "n_seeds": s["n"],
            "accept_rate": s["accept"],
            "smoothed_mAP": s["smoothed"],
            "smoothed_std": s["smoothed_std"],
            "final_mAP": s["final"],
            "final_optim_steps": s["steps"],
        })
    return pd.DataFrame(rows)


def ablation_pair_table(
    pairings: Sequence[Tuple[str, str, str]],
    *,
    project_root: Optional[Path] = None,
    tail_k: int = 5,
) -> pd.DataFrame:
    """Side-by-side stats for each (label, baseline, ablated) triple."""
    rows: List[Dict] = []
    for label, base_v, abl_v in pairings:
        b = _stats(base_v, project_root=project_root, tail_k=tail_k)
        a = _stats(abl_v, project_root=project_root, tail_k=tail_k)
        rows.append({
            "pair": label,
            "baseline_variant": base_v,
            "ablated_variant": abl_v,
            "baseline_accept": b["accept"],
            "ablated_accept": a["accept"],
            "baseline_smoothed": b["smoothed"],
            "ablated_smoothed": a["smoothed"],
            "delta_smoothed": a["smoothed"] - b["smoothed"],
            "baseline_steps": b["steps"],
            "ablated_steps": a["steps"],
        })
    return pd.DataFrame(rows)


def iso_accept_table(
    pairings: Sequence[Tuple[str, str]] = ISO_ACCEPT_PAIRINGS,
    *,
    project_root: Optional[Path] = None,
    tail_k: int = 5,
) -> pd.DataFrame:
    """Filter <-> closest-random pairings with smoothed mAP and accept gap."""
    rows: List[Dict] = []
    for f_v, r_v in pairings:
        f = _stats(f_v, project_root=project_root, tail_k=tail_k)
        r = _stats(r_v, project_root=project_root, tail_k=tail_k)
        if not (f["n"] and r["n"]):
            continue
        rows.append({
            "filter_variant": f_v,
            "filter_label": label_for(f_v),
            "random_variant": r_v,
            "random_label": label_for(r_v),
            "manifest": manifest_for_variant(f_v),
            "filter_accept": f["accept"],
            "random_accept": r["accept"],
            "accept_gap": f["accept"] - r["accept"],
            "filter_smoothed": f["smoothed"],
            "filter_smoothed_std": f["smoothed_std"],
            "random_smoothed": r["smoothed"],
            "random_smoothed_std": r["smoothed_std"],
            "delta_smoothed": f["smoothed"] - r["smoothed"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# Per-domain tables
# =============================================================================

def _per_domain_smoothed(
    variant: str,
    dim: str = "stream_block",
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
    tail_k: int = 5,
) -> Optional[pd.Series]:
    """Per-bucket smoothed-tail-k mAP for a variant (average over seeds)."""
    sd = variant_seed_dirs(variant, seeds=seeds, project_root=project_root)
    if not sd:
        return None
    rows = []
    for rdir in sd.values():
        pdc = ah.load_per_domain_checkpoints(rdir)
        if pdc is None or pdc.empty:
            continue
        last_k = sorted(pdc["checkpoint_idx"].unique())[-tail_k:]
        sub = pdc[(pdc["dimension"] == dim) & (pdc["checkpoint_idx"].isin(last_k))][
            ["bucket", "mAP"]
        ]
        rows.append(sub)
    if not rows:
        return None
    cat = pd.concat(rows, ignore_index=True)
    return cat.groupby("bucket")["mAP"].mean()


def per_domain_grid(
    variants: Sequence[str],
    *,
    dim: str = "stream_block",
    project_root: Optional[Path] = None,
    tail_k: int = 5,
) -> pd.DataFrame:
    """Wide-format ``block x variant`` grid of smoothed-tail-k mAP.

    The DataFrame's index is the bucket label and the columns are the
    variant labels (via `label_for`).  Missing (variant, bucket)
    cells are NaN.
    """
    series_by_label: Dict[str, pd.Series] = {}
    for v in variants:
        s = _per_domain_smoothed(v, dim=dim, project_root=project_root, tail_k=tail_k)
        if s is not None:
            series_by_label[label_for(v)] = s
    if not series_by_label:
        return pd.DataFrame()
    blocks = sorted(set().union(*[set(s.index) for s in series_by_label.values()]))
    df = pd.DataFrame({lab: [s.get(b, np.nan) for b in blocks]
                       for lab, s in series_by_label.items()},
                      index=pd.Index(blocks, name="block"))
    return df


def per_domain_summary(
    variants: Sequence[str],
    *,
    dim: str = "stream_block",
    project_root: Optional[Path] = None,
    tail_k: int = 5,
) -> pd.DataFrame:
    """Balanced (mean) and worst-bucket mAP per variant."""
    grid = per_domain_grid(variants, dim=dim,
                           project_root=project_root, tail_k=tail_k)
    if grid.empty:
        return pd.DataFrame()
    rows = []
    for lab in grid.columns:
        col = grid[lab].dropna()
        if col.empty:
            continue
        rows.append({
            "variant_label": lab,
            "balanced_mAP": float(col.mean()),
            "worst_block_mAP": float(col.min()),
            "worst_block": col.idxmin(),
            "best_block_mAP": float(col.max()),
            "best_block": col.idxmax(),
            "n_blocks": int(col.count()),
        })
    return pd.DataFrame(rows)


def per_domain_delta_grid(
    pairings: Sequence[Tuple[str, str]] = ISO_ACCEPT_PAIRINGS,
    *,
    dim: str = "stream_block",
    project_root: Optional[Path] = None,
    tail_k: int = 5,
) -> pd.DataFrame:
    """``block x pair`` grid of (filter mAP - random mAP) per block."""
    columns: Dict[str, pd.Series] = {}
    for f_v, r_v in pairings:
        f = _per_domain_smoothed(f_v, dim=dim, project_root=project_root, tail_k=tail_k)
        r = _per_domain_smoothed(r_v, dim=dim, project_root=project_root, tail_k=tail_k)
        if f is None or r is None:
            continue
        idx = sorted(set(f.index).union(r.index))
        columns[f"{label_for(f_v)} - {label_for(r_v)}"] = pd.Series(
            {b: f.get(b, np.nan) - r.get(b, np.nan) for b in idx}
        )
    if not columns:
        return pd.DataFrame()
    blocks = sorted(set().union(*[set(c.index) for c in columns.values()]))
    df = pd.DataFrame({k: [v.get(b, np.nan) for b in blocks]
                       for k, v in columns.items()},
                      index=pd.Index(blocks, name="block"))
    return df


def per_domain_delta_summary(
    pairings: Sequence[Tuple[str, str]] = ISO_ACCEPT_PAIRINGS,
    *,
    dim: str = "stream_block",
    project_root: Optional[Path] = None,
    tail_k: int = 5,
) -> pd.DataFrame:
    """Summary of the per-block deltas: mean / max / min / sign-counts per pair."""
    grid = per_domain_delta_grid(pairings, dim=dim,
                                 project_root=project_root, tail_k=tail_k)
    if grid.empty:
        return pd.DataFrame()
    rows = []
    for lab in grid.columns:
        col = grid[lab].dropna()
        if col.empty:
            continue
        rows.append({
            "comparison": lab,
            "mean_delta": float(col.mean()),
            "max_delta": float(col.max()),
            "argmax_block": col.idxmax(),
            "min_delta": float(col.min()),
            "argmin_block": col.idxmin(),
            "n_pos": int((col > 0).sum()),
            "n_total": int(col.count()),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Per-block accept-rate routing
# =============================================================================

def per_block_routing(
    variants: Sequence[str],
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Wide-format ``block x variant`` accept-rate grid (mean over seeds).

    Only emits a column for variants whose manifest has a defined
    ``ordering.block_order`` (i.e. curated-style block partitioned
    streams).  Temporal manifests stream chronologically and have no
    block grouping; they return no rows here.
    """
    project_root = project_root or ah.find_project_root()
    rates: Dict[str, Dict[str, float]] = {}
    block_order: List[str] = []
    for v in variants:
        sd = variant_seed_dirs(v, seeds=seeds, project_root=project_root)
        if not sd:
            continue
        per_seed = []
        man_used = None
        for rdir in sd.values():
            cfg = ah.load_run_config(rdir)
            man = ah.load_manifest(project_root, cfg.get("manifest_path") if cfg else None)
            if man is None:
                continue
            enr = ah.load_enriched_streaming_decisions(rdir, project_root, zod_root=None)
            if enr.empty:
                continue
            pb = ah.per_block_accept_rate(enr, man, bootstrap_frames=0)
            if not pb.empty:
                per_seed.append(pb)
                man_used = man
        if not per_seed:
            continue
        cat = pd.concat(per_seed, ignore_index=True)
        agg = cat.groupby("block_label", as_index=False)["accept_rate"].mean()
        rates[label_for(v)] = dict(zip(agg["block_label"], agg["accept_rate"]))
        if not block_order and man_used is not None:
            man_order = (man_used.get("ordering", {}) or {}).get("block_order", [])
            block_order = (
                [b for b in man_order if b in rates[label_for(v)]] +
                sorted(b for b in rates[label_for(v)] if b not in man_order)
            )
    if not rates:
        return pd.DataFrame()
    df = pd.DataFrame({lab: [rates[lab].get(b, np.nan) for b in block_order]
                       for lab in rates},
                      index=pd.Index(block_order, name="block"))
    return df


# =============================================================================
# Trajectory data (per-block / overall mAP through the stream)
# =============================================================================

def per_domain_trajectory(
    variant: str,
    blocks: Sequence[str],
    *,
    dim: str = "stream_block",
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Tidy long-format per-block mAP trajectory aggregated over seeds.

    Columns: ``checkpoint_idx, items_processed, optimizer_steps, bucket,
    mAP, mAP_std, n``.  ``mAP`` is the cross-seed mean and ``mAP_std``
    the cross-seed sample standard deviation (NaN if only one seed
    contributed at that checkpoint); ``n`` is the seed count.
    """
    sd = variant_seed_dirs(variant, seeds=seeds, project_root=project_root)
    parts = []
    for rdir in sd.values():
        pdc = ah.load_per_domain_checkpoints(rdir)
        if pdc is None or pdc.empty:
            continue
        sub = pdc[(pdc["dimension"] == dim) & (pdc["bucket"].isin(list(blocks)))]
        keep_cols = [c for c in
                     ("checkpoint_idx", "items_processed", "optimizer_steps",
                      "bucket", "mAP")
                     if c in sub.columns]
        parts.append(sub[keep_cols])
    if not parts:
        return pd.DataFrame()
    cat = pd.concat(parts, ignore_index=True)
    group_cols = [c for c in
                  ("checkpoint_idx", "items_processed", "optimizer_steps", "bucket")
                  if c in cat.columns]
    return cat.groupby(group_cols, as_index=False)["mAP"].agg(
        mAP="mean", mAP_std="std", n="count"
    )


def mAP_trajectory(
    variants: Sequence[str],
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Tidy long-format overall-mAP trajectory across variants.

    Columns: ``variant, label, items_processed, optimizer_steps, mAP, mAP_std, n``.
    """
    rows = []
    for v in variants:
        sd = variant_seed_dirs(v, seeds=seeds, project_root=project_root)
        per_seed = []
        for rdir in sd.values():
            ck = ah.read_csv(rdir / "checkpoints.csv")
            if ck is None or ck.empty:
                continue
            keep = [c for c in
                    ("items_processed", "optimizer_steps", "mAP")
                    if c in ck.columns]
            per_seed.append(ck[keep])
        if not per_seed:
            continue
        cat = pd.concat(per_seed, ignore_index=True)
        group_cols = [c for c in ("items_processed", "optimizer_steps") if c in cat.columns]
        agg = cat.groupby(group_cols, as_index=False)["mAP"].agg(
            mAP="mean", mAP_std="std", n="count"
        )
        agg.insert(0, "label", label_for(v))
        agg.insert(0, "variant", v)
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def rolling_accept_rate_per_variant(
    variant: str,
    *,
    project_root: Optional[Path] = None,
    seed: Optional[int] = 42,
    window: int = 1000,
) -> pd.DataFrame:
    """Rolling accept rate along the stream for a single variant/seed.

    Returns columns ``global_idx, accept_rate``; uses the actual decisions
    log so post-bootstrap items are indexed naturally.
    """
    rdir = latest_seed_dir(variant, seed, project_root=project_root)
    if rdir is None:
        return pd.DataFrame()
    dec = ah.read_csv(rdir / "decisions.csv")
    if dec is None or dec.empty:
        return pd.DataFrame()
    d = dec.sort_values("global_idx").copy()
    d["accept"] = (d["action"] == "accept").astype(float)
    win = max(50, min(window, max(5, len(d) // 10)))
    d["rolling_accept_rate"] = d["accept"].rolling(win, min_periods=1).mean()
    return d[["global_idx", "rolling_accept_rate"]].rename(
        columns={"rolling_accept_rate": "accept_rate"}
    )


def windowed_accept_rate_aggregated(
    variant: str,
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
    window: int = 1000,
) -> pd.DataFrame:
    """Mean +/- std accept rate in fixed-size windows, averaged across seeds.

    For each seed, bin the post-bootstrap decisions into non-overlapping
    ``window``-frame buckets and compute the per-bucket accept rate; then
    average across seeds at each bucket.  Returns columns ``items_start,
    accept_rate_mean, accept_rate_std, n_seeds``.
    """
    sd = variant_seed_dirs(variant, seeds=seeds, project_root=project_root)
    rows = []
    for seed, rdir in sd.items():
        dec = ah.read_csv(rdir / "decisions.csv")
        if dec is None or dec.empty:
            continue
        d = dec.sort_values("global_idx").copy()
        d["is_accept"] = (d["action"].astype(str) == "accept").astype(int)
        d["window"] = d["global_idx"] // int(window)
        g = d.groupby("window")["is_accept"].agg(["sum", "count"]).reset_index()
        g["accept_rate"] = g["sum"] / g["count"].clip(lower=1)
        g["items_start"] = g["window"] * int(window)
        g["seed"] = int(seed)
        rows.append(g[["seed", "items_start", "accept_rate"]])
    if not rows:
        return pd.DataFrame(
            columns=["items_start", "accept_rate_mean", "accept_rate_std", "n_seeds"]
        )
    long = pd.concat(rows, ignore_index=True)
    grp = long.groupby("items_start")["accept_rate"]
    out = pd.DataFrame({
        "accept_rate_mean": grp.mean(),
        "accept_rate_std": grp.std(ddof=0),
        "n_seeds": grp.count(),
    }).reset_index().sort_values("items_start").reset_index(drop=True)
    return out


def stream_composition(
    manifest: Mapping,
    *,
    bootstrap_frames: int,
    fields: Sequence[str] = ("time_of_day", "road_condition"),
    window: int = 1000,
    field_orders: Optional[Mapping[str, Sequence[str]]] = None,
    field_derivers: Optional[Mapping[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """Per-window stacked-area composition of the post-bootstrap stream.

    For each ``field`` in ``fields``, build a wide-format DataFrame
    whose index is ``items_start`` (window left edge) and columns are
    the field's values, holding the *fraction* of frames in that
    window with each value.

    Args:
        fields: per-frame metadata field names to summarise.  Each is
            looked up directly with ``frame.get(field)`` unless a
            matching entry in ``field_derivers`` is provided.
        field_orders: ``{field: ordered_categories}`` -- pins the column
            ordering (which drives the stacked-area layering).
        field_derivers: optional ``{field: callable(frame) -> str}``
            map for computed fields (e.g. a 5-bucket ``weather``
            derived from ``scraped_weather`` + ``road_condition``).
            Takes precedence over the raw lookup for that field.

    Missing values are dropped before summarising.
    """
    train = [f for f in manifest.get("frames", []) if f.get("split") == "train"]
    stream_frames = train[int(bootstrap_frames):]
    derivers = dict(field_derivers or {})

    def _value(field: str, frame: Mapping):
        deriver = derivers.get(field)
        return deriver(frame) if deriver is not None else frame.get(field)

    base = pd.DataFrame({
        "global_idx": range(len(stream_frames)),
        **{
            field: [_value(field, f) for f in stream_frames]
            for field in fields
        },
    })
    base["window"] = base["global_idx"] // int(window)
    base["items_start"] = base["window"] * int(window)
    out: Dict[str, pd.DataFrame] = {}
    field_orders = dict(field_orders or {})
    for field in fields:
        sub = base.dropna(subset=[field])
        if sub.empty:
            out[field] = pd.DataFrame()
            continue
        counts = (sub.groupby("items_start")[field]
                     .value_counts().unstack(fill_value=0))
        frac = counts.div(counts.sum(axis=1).clip(lower=1), axis=0)
        order = list(field_orders.get(field) or sorted(frac.columns))
        for col in order:
            if col not in frac.columns:
                frac[col] = 0.0
        # Place known values first, then any extras (rare manifests).
        extras = [c for c in frac.columns if c not in order]
        out[field] = frac[order + extras]
    return out


def block_boundaries_and_midpoints(
    manifest: Optional[Mapping],
    *,
    bootstrap_frames: int = 0,
) -> Tuple[List[int], List[Tuple[int, str]]]:
    """Return (boundaries, midpoints) along post-bootstrap stream coordinates.

    ``boundaries[k]`` is the start index of block k (so ``boundaries[1:]``
    holds the inter-block transitions).  ``midpoints[k] = (mid_x, label)``
    is useful for placing block-number annotations along the top of a
    plot.
    """
    if not manifest:
        return [], []
    ordering = manifest.get("ordering") or {}
    block_order = list(ordering.get("block_order") or [])
    block_sizes = ordering.get("block_sizes") or {}
    if not block_order or not block_sizes:
        return [], []
    boundaries: List[int] = []
    mids: List[Tuple[int, str]] = []
    pos = 0
    for b in block_order:
        size = int(block_sizes.get(b, 0))
        boundaries.append(pos)
        mids.append((pos + size // 2, b))
        pos += size
    boundaries.append(pos)  # right edge of last block
    return boundaries, mids


# =============================================================================
# Per-class AP (long-format trajectory and end-of-stream grids)
# =============================================================================

# Object classes the streaming runs evaluate.  Ordering controls plot
# ordering; matches the manifest's `target_classes` configuration.
TARGET_CLASSES: List[str] = ["Vehicle", "Pedestrian", "VulnerableVehicle"]


def _per_class_smoothed(
    variant: str,
    *,
    classes: Sequence[str] = TARGET_CLASSES,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
    tail_k: int = 5,
) -> Optional[pd.Series]:
    """Per-class smoothed-tail-k AP for a variant (mean across seeds).

    Reads the per-class ``AP_<class>`` columns from ``checkpoints.csv``.
    """
    sd = variant_seed_dirs(variant, seeds=seeds, project_root=project_root)
    if not sd:
        return None
    rows = []
    for rdir in sd.values():
        ck = ah.read_csv(rdir / "checkpoints.csv")
        if ck is None or ck.empty:
            continue
        cols = [f"AP_{c}" for c in classes if f"AP_{c}" in ck.columns]
        if not cols:
            continue
        tail = ck.tail(tail_k)[cols]
        rows.append(tail.mean())
    if not rows:
        return None
    return pd.concat(rows, axis=1).mean(axis=1)


def per_class_grid(
    variants: Sequence[str],
    *,
    classes: Sequence[str] = TARGET_CLASSES,
    project_root: Optional[Path] = None,
    tail_k: int = 5,
) -> pd.DataFrame:
    """Wide ``class x variant`` grid of smoothed end-of-stream per-class AP."""
    cols: Dict[str, pd.Series] = {}
    for v in variants:
        s = _per_class_smoothed(v, classes=classes,
                                project_root=project_root, tail_k=tail_k)
        if s is not None:
            cols[label_for(v)] = s
    if not cols:
        return pd.DataFrame()
    classes_list = [f"AP_{c}" for c in classes]
    df = pd.DataFrame({lab: [c.get(k, np.nan) for k in classes_list]
                       for lab, c in cols.items()},
                      index=pd.Index([c.replace("AP_", "") for c in classes_list],
                                     name="class"))
    return df


def per_class_trajectory(
    variant: str,
    *,
    classes: Sequence[str] = TARGET_CLASSES,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Long-format per-class AP trajectory through the stream (mean over seeds).

    Columns: ``items_processed, optimizer_steps, class, AP``.
    """
    sd = variant_seed_dirs(variant, seeds=seeds, project_root=project_root)
    parts = []
    for rdir in sd.values():
        ck = ah.read_csv(rdir / "checkpoints.csv")
        if ck is None or ck.empty:
            continue
        keep = [c for c in ("items_processed", "optimizer_steps") if c in ck.columns]
        ap_cols = [f"AP_{c}" for c in classes if f"AP_{c}" in ck.columns]
        if not ap_cols:
            continue
        long = ck[keep + ap_cols].melt(id_vars=keep, var_name="ap_col", value_name="AP")
        long["class"] = long["ap_col"].str.replace("AP_", "", regex=False)
        parts.append(long.drop(columns=["ap_col"]))
    if not parts:
        return pd.DataFrame()
    cat = pd.concat(parts, ignore_index=True)
    group_cols = [c for c in ("items_processed", "optimizer_steps", "class") if c in cat.columns]
    return cat.groupby(group_cols, as_index=False)["AP"].mean()


# =============================================================================
# Forgetting analysis (per-class first vs last quartile)
# =============================================================================

def forgetting_table(
    variants: Sequence[str],
    *,
    classes: Sequence[str] = TARGET_CLASSES,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
    n_bins: int = 4,
) -> pd.DataFrame:
    """Mean AP in the first vs the last quartile of the stream, per (variant, class).

    Negative ``delta`` (last - first) signals catastrophic-forgetting-like
    behavior on that class as the stream progresses.  Aggregated as the
    mean of the per-seed per-class deltas.
    """
    rows = []
    for v in variants:
        sd = variant_seed_dirs(v, seeds=seeds, project_root=project_root)
        if not sd:
            continue
        per_class_deltas: Dict[str, List[float]] = {c: [] for c in classes}
        per_class_early: Dict[str, List[float]] = {c: [] for c in classes}
        per_class_late: Dict[str, List[float]] = {c: [] for c in classes}
        for rdir in sd.values():
            ck = ah.read_csv(rdir / "checkpoints.csv")
            if ck is None or ck.empty or "items_processed" not in ck.columns:
                continue
            ap_cols = [f"AP_{c}" for c in classes if f"AP_{c}" in ck.columns]
            if not ap_cols:
                continue
            ft = ah.forgetting_table(ck, ap_cols, n_bins=n_bins)
            if ft.empty:
                continue
            for col in ap_cols:
                cls = col.replace("AP_", "")
                per_class_deltas[cls].append(float(ft.loc[col, "delta"]))
                per_class_early[cls].append(float(ft.loc[col, "early"]))
                per_class_late[cls].append(float(ft.loc[col, "late"]))
        for cls in classes:
            if not per_class_deltas[cls]:
                continue
            rows.append({
                "variant": v,
                "label": label_for(v),
                "class": cls,
                "early": float(np.mean(per_class_early[cls])),
                "late": float(np.mean(per_class_late[cls])),
                "delta": float(np.mean(per_class_deltas[cls])),
                "n_seeds": len(per_class_deltas[cls]),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Refresh-segment accept rate (within-block decay diagnostic)
# =============================================================================

def refresh_segment_table(
    variants: Sequence[str],
    *,
    project_root: Optional[Path] = None,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Accept rate inside each inter-refresh segment for adaptive variants.

    Reads `decisions.csv` and `refreshes.csv` for one seed (default 42)
    per variant, and returns columns ``variant, label, refresh_idx,
    segment_start, segment_end, n_frames, n_accepts, accept_rate``.
    """
    project_root = project_root or ah.find_project_root()
    rows = []
    for v in variants:
        rdir = latest_seed_dir(v, seed, project_root=project_root)
        if rdir is None:
            continue
        cfg = ah.load_run_config(rdir)
        fam = ah.filter_mode(cfg)
        if fam not in {"window", "reservoir"}:
            continue
        dec = ah.read_csv(rdir / "decisions.csv")
        if dec is None or dec.empty:
            continue
        seg = ah.refresh_accept_rate_segments(dec, ah.load_refreshes(rdir))
        if seg.empty:
            continue
        seg = seg.copy()
        seg.insert(0, "label", label_for(v))
        seg.insert(0, "variant", v)
        rows.append(seg)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# Compute efficiency (steps to reach a target mAP)
# =============================================================================

def steps_to_reach_table(
    variants: Sequence[str],
    targets: Sequence[float],
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
    x_col: str = "optimizer_steps",
) -> pd.DataFrame:
    """For each (variant, target), interpolate the smallest ``x`` at which
    seed-averaged mAP >= target.

    Useful for the "fewer steps to reach mAP X" claim.  Each target is
    reported in its own row.  When a variant never crosses the target,
    the value is ``NaN``.
    """
    rows = []
    for v in variants:
        sd = variant_seed_dirs(v, seeds=seeds, project_root=project_root)
        per_seed = []
        for rdir in sd.values():
            ck = ah.read_csv(rdir / "checkpoints.csv")
            if ck is None or ck.empty or x_col not in ck.columns or "mAP" not in ck.columns:
                continue
            per_seed.append(ck[[x_col, "mAP"]].copy())
        if not per_seed:
            continue
        cat = pd.concat(per_seed, ignore_index=True)
        agg = (cat.groupby(x_col, as_index=False)["mAP"].mean()
                  .sort_values(x_col).reset_index(drop=True))
        for tgt in targets:
            xs = agg[x_col].to_numpy(dtype=float)
            ys = agg["mAP"].to_numpy(dtype=float)
            cross = np.where(ys >= tgt)[0]
            if len(cross) == 0:
                rows.append({"variant": v, "label": label_for(v),
                             "target_mAP": tgt, x_col: float("nan"),
                             "max_mAP": float(ys.max())})
                continue
            i = int(cross[0])
            if i == 0 or ys[i] == tgt:
                x_at = float(xs[i])
            else:
                # Linear interpolate between the bracketing x values.
                x0, y0 = xs[i - 1], ys[i - 1]
                x1, y1 = xs[i], ys[i]
                if y1 == y0:
                    x_at = float(x1)
                else:
                    x_at = float(x0 + (tgt - y0) * (x1 - x0) / (y1 - y0))
            rows.append({"variant": v, "label": label_for(v),
                         "target_mAP": tgt, x_col: x_at,
                         "max_mAP": float(ys.max())})
    return pd.DataFrame(rows)


# =============================================================================
# Convenience: build the full set of streaming summary tables in one call
# =============================================================================

def build_summary_tables(
    *,
    project_root: Optional[Path] = None,
    tail_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Compute every standard summary table; returns a dict keyed by table name.

    Useful for ``analyze_runs.py`` or any one-shot regenerator.  Tables:

    - ``inventory``
    - ``iso_accept``
    - ``ablation_twoRef``
    - ``ablation_noBoot``
    - ``static_vs_adaptive``
    - ``per_domain_curated``  (block x variant grid)
    - ``per_domain_temporal``
    - ``per_domain_summary_curated``
    - ``per_domain_summary_temporal``
    - ``per_domain_delta``    (block x pairing grid)
    - ``per_domain_delta_summary``
    - ``per_block_routing``   (curated only)
    """
    project_root = project_root or ah.find_project_root()
    prime_registry(project_root=project_root)

    inv = inventory_table(project_root=project_root, tail_k=tail_k)

    cur_variants = [v for v in FEATURED_VARIANTS if manifest_for_variant(v) == "curated"]
    tmp_variants = [v for v in FEATURED_VARIANTS if manifest_for_variant(v) == "temporal"]

    return {
        "inventory": inv,
        "iso_accept": iso_accept_table(project_root=project_root, tail_k=tail_k),
        "ablation_twoRef": ablation_pair_table(
            ABLATION_PAIRINGS["twoRef"], project_root=project_root, tail_k=tail_k),
        "ablation_noBoot": ablation_pair_table(
            ABLATION_PAIRINGS["noBoot"], project_root=project_root, tail_k=tail_k),
        "static_vs_adaptive": ablation_pair_table(
            ABLATION_PAIRINGS["static_vs_adaptive"],
            project_root=project_root, tail_k=tail_k),
        "per_domain_curated": per_domain_grid(
            cur_variants, project_root=project_root, tail_k=tail_k),
        "per_domain_temporal": per_domain_grid(
            tmp_variants, project_root=project_root, tail_k=tail_k),
        "per_domain_summary_curated": per_domain_summary(
            cur_variants, project_root=project_root, tail_k=tail_k),
        "per_domain_summary_temporal": per_domain_summary(
            tmp_variants, project_root=project_root, tail_k=tail_k),
        "per_domain_delta": per_domain_delta_grid(
            project_root=project_root, tail_k=tail_k),
        "per_domain_delta_summary": per_domain_delta_summary(
            project_root=project_root, tail_k=tail_k),
        "per_block_routing": per_block_routing(
            cur_variants, project_root=project_root),
        "per_class_curated": per_class_grid(
            cur_variants, project_root=project_root, tail_k=tail_k),
        "per_class_temporal": per_class_grid(
            tmp_variants, project_root=project_root, tail_k=tail_k),
        "forgetting_curated": forgetting_table(
            cur_variants, project_root=project_root),
    }
