"""
Federated-specific analysis tables.

Mirrors `streaming` for the federated pipeline; operates on the standard
layout::

    outputs/federated/<variant>/seed_<N>/<YYYY-mm-dd_HH-MM-SS>/

via the run-discovery and CSV-loading primitives in `runs`.

Public surface (the things the federated notebook actually calls):

    Registry / lookups
        FEATURED_VARIANTS              -- ordered list of canonical variants
        VARIANT_LABEL                  -- variant -> short display label
        VARIANT_FAMILY                 -- variant -> filter family
        ISO_ACCEPT_PAIRINGS            -- explicit filter <-> random pairs
        ABLATION_PAIRINGS              -- single <-> ablation pairs
        CLIENT_LABEL                   -- 0..3 -> "C<i> <description>"
        DOMAIN_BLOCK_FAMILY            -- block -> "familiar" | "novel"
        latest_seed_dir(variant, seed) -- newest run dir for a (variant, seed)
        variant_seed_dirs(variant)     -- {seed: run_dir} across all seeds
        manifest_for_variant(v)        -- "curated" / "curated_heavyLocal" / ...
        schedule_for_variant(v)        -- "default" / "heavyLocal" / ...

    Headline tables
        inventory_table(variants)
        ablation_pair_table(pairings)
        iso_accept_table(pairings)

    Per-client tables (federated-specific)
        per_client_accept_table(variants)
        novelty_routing_summary(variants)

    Per-block (validation-domain) tables
        per_domain_grid(variants, dim, k)
        per_domain_summary(variants, dim, k)
        per_domain_delta_grid(pairings, dim, k)
        per_domain_delta_summary(pairings, dim, k)
        per_block_trajectory_delta(pairings)

    Trajectory / per-class / refresh
        mAP_trajectory(variants)
        per_class_grid(variants, classes, k)
        per_class_trajectory(variant)
        refresh_segment_table(variants)

    Convenience
        build_summary_tables(...)

All numeric helpers return tidy long-format pandas DataFrames so they
drop straight into figure-making code without further reshaping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import runs as ah


# =============================================================================
# Variant registry
# =============================================================================

# Ordered list of variants featured in the federated write-up.  The
# order is also the row order for tables and the legend order for
# figures.  Mirrors the streaming registry but for the federated grid.
FEATURED_VARIANTS: List[str] = [
    # Phase 1A - filter grid on cityday_curated (default schedule)
    "fed_no_filter_cityday_curated",
    "fed_static_p20_cityday_curated",
    "fed_adaptive_window_p20_cityday_curated",
    "fed_adaptive_window_p20_twoRef_cityday_curated",
    "fed_adaptive_reservoir_p20_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    # Phase 1B - iso-accept random partners
    "fed_random_p12_cityday_curated",
    "fed_random_p15_cityday_curated",
    "fed_random_p18_cityday_curated",
    "fed_random_p77_cityday_curated",
    # Phase 2 Y - tighter accept ablation
    "fed_adaptive_window_p10_cityday_curated",
    "fed_adaptive_reservoir_p10_cityday_curated",
    "fed_adaptive_reservoir_p10_twoRef_cityday_curated",
    # Phase 2 Z - heavier local schedule
    "fed_no_filter_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
    "fed_random_p15_cityday_curated_heavyLocal",
    "fed_random_p18_cityday_curated_heavyLocal",
]

# Short display labels.  Missing entries fall back to the variant name.
VARIANT_LABEL: Dict[str, str] = {
    # Phase 1A
    "fed_no_filter_cityday_curated":                              "none",
    "fed_static_p20_cityday_curated":                             "static p20",
    "fed_adaptive_window_p20_cityday_curated":                    "window p20",
    "fed_adaptive_window_p20_twoRef_cityday_curated":             "window p20 twoRef",
    "fed_adaptive_reservoir_p20_cityday_curated":                 "reservoir p20",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated":          "reservoir p20 twoRef",
    # Phase 1B
    "fed_random_p12_cityday_curated":                             "random p12",
    "fed_random_p15_cityday_curated":                             "random p15",
    "fed_random_p18_cityday_curated":                             "random p18",
    "fed_random_p77_cityday_curated":                             "random p77",
    # Phase 2 Y
    "fed_adaptive_window_p10_cityday_curated":                    "window p10",
    "fed_adaptive_reservoir_p10_cityday_curated":                 "reservoir p10",
    "fed_adaptive_reservoir_p10_twoRef_cityday_curated":          "reservoir p10 twoRef",
    # Phase 2 Z (heavyLocal)
    "fed_no_filter_cityday_curated_heavyLocal":                   "none (HL)",
    "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal":      "reservoir p20 (HL)",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal":
                                                                  "reservoir p20 twoRef (HL)",
    "fed_random_p15_cityday_curated_heavyLocal":                  "random p15 (HL)",
    "fed_random_p18_cityday_curated_heavyLocal":                  "random p18 (HL)",
}


def label_for(variant: str) -> str:
    """Canonical short label for a variant (fallback: variant)."""
    return VARIANT_LABEL.get(variant, variant)


def manifest_for_variant(variant: str) -> str:
    """Return ``"curated"``, ``"temporal"``, ... from the variant name."""
    if "cityday_temporal" in variant:
        return "temporal"
    if "cityday_curated" in variant:
        return "curated"
    return "unknown"


def schedule_for_variant(variant: str) -> str:
    """Return ``"heavyLocal"`` for Phase Z variants, else ``"default"``."""
    return "heavyLocal" if variant.endswith("_heavyLocal") else "default"


def family_for_variant(variant: str, project_root: Optional[Path] = None) -> str:
    """Filter family from the run config; ``"unknown"`` when not discoverable."""
    rdir = latest_seed_dir(variant, seed=None, project_root=project_root)
    if rdir is None:
        # Heuristic from name when no run dir is available.
        if "no_filter" in variant:
            return "none"
        if "_random_" in variant:
            return "random"
        if "_static_" in variant:
            return "static"
        if "_window_" in variant:
            return "window"
        if "_reservoir_" in variant:
            return "reservoir"
        return "unknown"
    return ah.filter_mode(ah.load_run_config(rdir))


# Per-variant filter family (populated by prime_registry).
VARIANT_FAMILY: Dict[str, str] = {}


# =============================================================================
# Iso-accept and ablation pairings
# =============================================================================

# Iso-accept pairings used in the federated write-up.  Each tuple is
# ``(filter_variant, random_variant)``.  Random fractions were chosen
# to match the empirical Phase 1A accept rates.
ISO_ACCEPT_PAIRINGS: List[Tuple[str, str]] = [
    # Phase 1A vs Phase 1B
    ("fed_static_p20_cityday_curated",                            "fed_random_p77_cityday_curated"),
    ("fed_adaptive_window_p20_cityday_curated",                   "fed_random_p12_cityday_curated"),
    ("fed_adaptive_window_p20_twoRef_cityday_curated",            "fed_random_p12_cityday_curated"),
    ("fed_adaptive_reservoir_p20_cityday_curated",                "fed_random_p18_cityday_curated"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_curated",         "fed_random_p15_cityday_curated"),
    # Phase 2 Z (heavyLocal) - random partners shown if they exist
    ("fed_adaptive_reservoir_p20_cityday_curated_heavyLocal",
     "fed_random_p18_cityday_curated_heavyLocal"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
     "fed_random_p15_cityday_curated_heavyLocal"),
]

# Ablation pairings: ``(label, baseline_variant, ablated_variant)``.
ABLATION_PAIRINGS: Dict[str, List[Tuple[str, str, str]]] = {
    "twoRef": [
        ("Win_p20",
         "fed_adaptive_window_p20_cityday_curated",
         "fed_adaptive_window_p20_twoRef_cityday_curated"),
        ("Res_p20",
         "fed_adaptive_reservoir_p20_cityday_curated",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated"),
        ("Res_p10",
         "fed_adaptive_reservoir_p10_cityday_curated",
         "fed_adaptive_reservoir_p10_twoRef_cityday_curated"),
        ("Res_p20  (HL)",
         "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal"),
    ],
    "tighter_accept": [
        ("Win  p20 -> p10",
         "fed_adaptive_window_p20_cityday_curated",
         "fed_adaptive_window_p10_cityday_curated"),
        ("Res  p20 -> p10",
         "fed_adaptive_reservoir_p20_cityday_curated",
         "fed_adaptive_reservoir_p10_cityday_curated"),
        ("Res  p20 -> p10  (twoRef)",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
         "fed_adaptive_reservoir_p10_twoRef_cityday_curated"),
    ],
    "heavyLocal": [
        ("none",
         "fed_no_filter_cityday_curated",
         "fed_no_filter_cityday_curated_heavyLocal"),
        ("Res_p20",
         "fed_adaptive_reservoir_p20_cityday_curated",
         "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal"),
        ("Res_p20  twoRef",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal"),
    ],
    "static_vs_adaptive": [
        ("p20  (curated)",
         "fed_static_p20_cityday_curated",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated"),
    ],
}


# =============================================================================
# Federated client / domain bookkeeping
# =============================================================================

# Display labels for the four `domain_aligned` clients we use.  These
# descriptions match the `domain_client_groups` in the curated configs.
CLIENT_LABEL: Dict[int, str] = {
    0: "C0 city_day_familiar",
    1: "C1 city_day_novel",
    2: "C2 urban_arterial",
    3: "C3 out_of_city",
}


# Mapping from `stream_block` bucket name to the higher-level grouping
# used for the routing-vs-domain story.  "familiar" = the bootstrap mode
# (city_day_clear/cloudy), "novel" = everything else.
DOMAIN_BLOCK_FAMILY: Dict[str, str] = {
    "city_day_clear":             "familiar",
    "city_day_cloudy":            "familiar",
    "city_day_rain_wet":          "novel",
    "city_day_snow":              "novel",
    "city_twilight":              "novel",
    "city_night":                 "novel",
    "arterial-urban_day":         "novel",
    "arterial-urban_twi-night":   "novel",
    "highway_day":                "novel",
    "highway_twi-night":          "novel",
    "arterial-rural_day":         "novel",
    "arterial-rural_twi-night":   "novel",
    "smaller-rural_all":          "novel",
}


def block_family(block: str) -> str:
    """Return ``"familiar"``, ``"novel"`` or ``"unknown"`` for a block name."""
    return DOMAIN_BLOCK_FAMILY.get(block, "unknown")


# =============================================================================
# Run-dir resolution
# =============================================================================

def latest_seed_dir(
    variant: str,
    seed: Optional[int] = None,
    *,
    project_root: Optional[Path] = None,
) -> Optional[Path]:
    """Return the most-recent federated run directory for a (variant, seed).

    When ``seed`` is ``None`` the lowest-numbered seed available is used,
    so callers that just need *any* run dir get a deterministic answer.
    """
    root = (project_root or ah.find_project_root()) / "outputs" / "federated" / variant
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


def num_clients_for_run(run_dir: Path) -> int:
    """Number of clients for a federated run, derived from rounds.csv columns."""
    rd = ah.read_csv(run_dir / "rounds.csv")
    if rd is None or rd.empty:
        cfg = ah.load_run_config(run_dir)
        return int(cfg.get("num_clients", 0)) if cfg else 0
    return sum(1 for c in rd.columns if c.startswith("client_") and c.endswith("_items"))


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
    """Headline statistics for a variant (mean across seeds).

    Returns dict with ``n, accept, accept_std, smoothed, smoothed_std,
    final, items, opt_steps, n_rounds``.  ``smoothed`` averages the last
    ``tail_k`` overall mAP rows from ``rounds.csv``.  ``items`` and
    ``opt_steps`` are the cumulative totals at the final round.
    """
    sd = variant_seed_dirs(variant, seeds=seeds, project_root=project_root)
    accs, smoothed, finals, items, steps, n_rounds = [], [], [], [], [], []
    for run in sd.values():
        rd = ah.read_csv(run / "rounds.csv")
        if rd is None or rd.empty:
            continue
        item_cols = [c for c in rd.columns if c.startswith("client_") and c.endswith("_items")]
        acc_cols = [c for c in rd.columns if c.startswith("client_") and c.endswith("_accepted")]
        n_items = float(rd[item_cols].sum().sum()) if item_cols else 0.0
        n_acc = float(rd[acc_cols].sum().sum()) if acc_cols else 0.0
        if n_items > 0:
            accs.append(n_acc / n_items)
        valid = rd.dropna(subset=["mAP"]) if "mAP" in rd.columns else rd
        if not valid.empty:
            finals.append(float(valid["mAP"].iloc[-1]))
            tail = valid["mAP"].iloc[-tail_k:]
            smoothed.append(float(tail.mean()))
        if "items_processed_total" in rd.columns:
            items.append(int(rd["items_processed_total"].iloc[-1]))
        if "optimizer_steps_total" in rd.columns:
            steps.append(int(rd["optimizer_steps_total"].iloc[-1]))
        n_rounds.append(int(len(rd)))
    return {
        "n": len(sd),
        "accept": float(np.mean(accs)) if accs else float("nan"),
        "accept_std": float(np.std(accs)) if len(accs) > 1 else float("nan"),
        "smoothed": float(np.mean(smoothed)) if smoothed else float("nan"),
        "smoothed_std": float(np.std(smoothed)) if len(smoothed) > 1 else float("nan"),
        "final": float(np.mean(finals)) if finals else float("nan"),
        "items": int(np.mean(items)) if items else None,
        "opt_steps": int(np.mean(steps)) if steps else None,
        "n_rounds": int(np.mean(n_rounds)) if n_rounds else None,
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
    """One row per variant: accept rate, smoothed tail-k mAP, total items, etc."""
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
            "schedule": schedule_for_variant(v),
            "family": family_for_variant(v, project_root=project_root),
            "n_seeds": s["n"],
            "n_rounds": s["n_rounds"],
            "accept_rate": s["accept"],
            "accept_rate_std": s["accept_std"],
            "smoothed_mAP": s["smoothed"],
            "smoothed_std": s["smoothed_std"],
            "final_mAP": s["final"],
            "items_processed": s["items"],
            "optimizer_steps": s["opt_steps"],
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
            "delta_smoothed": (a["smoothed"] - b["smoothed"])
                              if not (np.isnan(a["smoothed"]) or np.isnan(b["smoothed"]))
                              else float("nan"),
            "baseline_n_seeds": b["n"],
            "ablated_n_seeds": a["n"],
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
            "schedule": schedule_for_variant(f_v),
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
# Per-client tables (federated-specific)
# =============================================================================

def per_client_accept_table(
    variants: Optional[Sequence[str]] = None,
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Per-(variant, client) accept rate, mean across seeds.

    Columns: ``variant, label, family, schedule, client, client_label,
    n_seeds, items, accepted, accept_rate``.
    """
    variants = list(variants) if variants is not None else list(FEATURED_VARIANTS)
    rows: List[Dict] = []
    for v in variants:
        sd = variant_seed_dirs(v, seeds=seeds, project_root=project_root)
        if not sd:
            continue
        per_client_acc: Dict[int, List[float]] = {}
        per_client_items: Dict[int, List[float]] = {}
        per_client_accepted: Dict[int, List[float]] = {}
        for rdir in sd.values():
            rd = ah.read_csv(rdir / "rounds.csv")
            if rd is None or rd.empty:
                continue
            item_cols = [c for c in rd.columns if c.startswith("client_") and c.endswith("_items")]
            for c in item_cols:
                cid = int(c.split("_", 2)[1])
                items = float(rd[c].sum())
                acc = float(rd[f"client_{cid}_accepted"].sum())
                per_client_items.setdefault(cid, []).append(items)
                per_client_accepted.setdefault(cid, []).append(acc)
                if items > 0:
                    per_client_acc.setdefault(cid, []).append(acc / items)
        family = family_for_variant(v, project_root=project_root)
        schedule = schedule_for_variant(v)
        for cid in sorted(per_client_items.keys()):
            rates = per_client_acc.get(cid, [])
            rows.append({
                "variant": v,
                "label": label_for(v),
                "family": family,
                "schedule": schedule,
                "client": cid,
                "client_label": CLIENT_LABEL.get(cid, f"C{cid}"),
                "n_seeds": len(rates),
                "items": float(np.mean(per_client_items[cid])),
                "accepted": float(np.mean(per_client_accepted[cid])),
                "accept_rate": float(np.mean(rates)) if rates else float("nan"),
                "accept_rate_std": (float(np.std(rates))
                                    if len(rates) > 1 else float("nan")),
            })
    return pd.DataFrame(rows)


def novelty_routing_summary(
    variants: Optional[Sequence[str]] = None,
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """One row per variant: per-client accept rates + novelty ratio.

    ``novelty_ratio = mean(C1, C2, C3 accept) / C0_accept``.  >1 means
    the filter routes more compute to novel-domain clients than to the
    familiar client; ~=1 means flat routing (e.g. random).
    """
    long = per_client_accept_table(variants, project_root=project_root, seeds=seeds)
    if long.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    for variant, sub in long.groupby("variant", sort=False):
        client_rates = dict(zip(sub["client"], sub["accept_rate"]))
        c0 = client_rates.get(0, float("nan"))
        novel_ids = [c for c in client_rates if c != 0]
        novel_mean = (float(np.mean([client_rates[c] for c in novel_ids]))
                      if novel_ids else float("nan"))
        novelty_ratio = (novel_mean / c0) if (c0 and not np.isnan(c0) and c0 > 0
                                              and not np.isnan(novel_mean)) else float("nan")
        rows.append({
            "variant": variant,
            "label": label_for(variant),
            "family": sub["family"].iloc[0],
            "schedule": sub["schedule"].iloc[0],
            **{f"C{cid}_accept": client_rates.get(cid, float("nan"))
               for cid in sorted(client_rates.keys())},
            "novel_mean_accept": novel_mean,
            "novelty_ratio": novelty_ratio,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Per-block (validation-domain) tables
# =============================================================================

def _per_domain_smoothed(
    variant: str,
    dim: str = "stream_block",
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
    tail_k: int = 5,
) -> Optional[pd.Series]:
    """Per-bucket smoothed-tail-k mAP for a variant (mean over seeds)."""
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
    """Wide-format ``block x variant`` grid of smoothed-tail-k mAP."""
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
    grid = per_domain_grid(variants, dim=dim, project_root=project_root, tail_k=tail_k)
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
    """Summary of the per-block deltas: stratified by familiar / novel."""
    grid = per_domain_delta_grid(pairings, dim=dim,
                                 project_root=project_root, tail_k=tail_k)
    if grid.empty:
        return pd.DataFrame()
    rows = []
    for lab in grid.columns:
        col = grid[lab].dropna()
        if col.empty:
            continue
        fam_blocks = [b for b in col.index if block_family(b) == "familiar"]
        nov_blocks = [b for b in col.index if block_family(b) == "novel"]
        fam_vals = col.loc[fam_blocks] if fam_blocks else pd.Series([], dtype=float)
        nov_vals = col.loc[nov_blocks] if nov_blocks else pd.Series([], dtype=float)
        rows.append({
            "comparison": lab,
            "n_blocks": int(col.count()),
            "mean_delta": float(col.mean()),
            "max_delta": float(col.max()),
            "argmax_block": col.idxmax(),
            "min_delta": float(col.min()),
            "argmin_block": col.idxmin(),
            "n_pos": int((col > 0).sum()),
            "novel_mean_delta": float(nov_vals.mean()) if len(nov_vals) else float("nan"),
            "novel_n_pos": int((nov_vals > 0).sum()) if len(nov_vals) else 0,
            "novel_n_total": len(nov_vals),
            "familiar_mean_delta": float(fam_vals.mean()) if len(fam_vals) else float("nan"),
            "familiar_n_pos": int((fam_vals > 0).sum()) if len(fam_vals) else 0,
            "familiar_n_total": len(fam_vals),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Per-block trajectory deltas (the "did the filter learn novel blocks faster?"
# diagnostic that uses the whole training history rather than just the tail)
# =============================================================================

def _per_block_trajectory(
    variant: str,
    dim: str = "stream_block",
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> Dict[str, List[float]]:
    """``{block: [mAP per checkpoint, mean across seeds]}`` for a variant."""
    sd = variant_seed_dirs(variant, seeds=seeds, project_root=project_root)
    per_seed: Dict[str, List[List[float]]] = {}
    for rdir in sd.values():
        pdc = ah.load_per_domain_checkpoints(rdir)
        if pdc is None or pdc.empty:
            continue
        sub = pdc[pdc["dimension"] == dim].sort_values("checkpoint_idx")
        if sub.empty:
            continue
        for block, grp in sub.groupby("bucket"):
            per_seed.setdefault(block, []).append(grp["mAP"].tolist())
    out: Dict[str, List[float]] = {}
    for block, lists in per_seed.items():
        n = min(len(l) for l in lists)
        if n == 0:
            continue
        out[block] = [float(np.mean([l[i] for l in lists])) for i in range(n)]
    return out


def per_block_trajectory_delta(
    pairings: Sequence[Tuple[str, str]] = ISO_ACCEPT_PAIRINGS,
    *,
    dim: str = "stream_block",
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Per-block ``cum_avg_delta`` and ``final_delta`` across (filter, random) pairs.

    For each pair and each block we compute, after averaging the mAP
    trajectory across seeds:

    - ``cum_avg_delta``: mean over rounds of ``filter[i] - random[i]``.
      Captures whether the filter held an advantage *throughout* training.
    - ``auc_delta``: sum over rounds (=cum_avg_delta * n_rounds).
    - ``final_delta``: filter[-1] - random[-1] (last round only).

    Returns long-format columns ``filter_variant, filter_label,
    random_variant, random_label, block, family, n_rounds, cum_avg_delta,
    auc_delta, final_delta``.
    """
    rows = []
    for f_v, r_v in pairings:
        ft = _per_block_trajectory(f_v, dim=dim, project_root=project_root, seeds=seeds)
        rt = _per_block_trajectory(r_v, dim=dim, project_root=project_root, seeds=seeds)
        if not ft or not rt:
            continue
        for block in sorted(ft.keys() & rt.keys()):
            n = min(len(ft[block]), len(rt[block]))
            if n == 0:
                continue
            deltas = [ft[block][i] - rt[block][i] for i in range(n)]
            rows.append({
                "filter_variant": f_v,
                "filter_label": label_for(f_v),
                "random_variant": r_v,
                "random_label": label_for(r_v),
                "block": block,
                "family": block_family(block),
                "n_rounds": n,
                "cum_avg_delta": float(np.mean(deltas)),
                "auc_delta": float(np.sum(deltas)),
                "final_delta": float(deltas[-1]),
            })
    return pd.DataFrame(rows)


# =============================================================================
# mAP / per-class trajectories
# =============================================================================

def mAP_trajectory(
    variants: Sequence[str],
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Tidy long-format overall-mAP trajectory across variants.

    Columns: ``variant, label, round, items_processed_total,
    optimizer_steps_total, mAP, mAP_std, n``.
    """
    rows = []
    for v in variants:
        sd = variant_seed_dirs(v, seeds=seeds, project_root=project_root)
        per_seed = []
        for rdir in sd.values():
            rd = ah.read_csv(rdir / "rounds.csv")
            if rd is None or rd.empty or "mAP" not in rd.columns:
                continue
            keep = [c for c in
                    ("round", "items_processed_total",
                     "optimizer_steps_total", "mAP")
                    if c in rd.columns]
            per_seed.append(rd[keep])
        if not per_seed:
            continue
        cat = pd.concat(per_seed, ignore_index=True)
        group_cols = [c for c in
                      ("round", "items_processed_total", "optimizer_steps_total")
                      if c in cat.columns]
        agg = cat.groupby(group_cols, as_index=False)["mAP"].agg(
            mAP="mean", mAP_std="std", n="count"
        )
        agg.insert(0, "label", label_for(v))
        agg.insert(0, "variant", v)
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def per_block_trajectory(
    variant: str,
    blocks: Sequence[str],
    *,
    dim: str = "stream_block",
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Long-format per-block mAP trajectory (mean over seeds).

    Columns: ``checkpoint_idx, items_processed, optimizer_steps, bucket, mAP``.
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
    return cat.groupby(group_cols, as_index=False)["mAP"].mean()


# =============================================================================
# Per-class AP
# =============================================================================

# Object classes the federated runs evaluate (matches streaming).
TARGET_CLASSES: List[str] = ["Vehicle", "Pedestrian", "VulnerableVehicle"]


def _per_class_smoothed(
    variant: str,
    *,
    classes: Sequence[str] = TARGET_CLASSES,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
    tail_k: int = 5,
) -> Optional[pd.Series]:
    """Per-class smoothed-tail-k AP for a variant (mean across seeds)."""
    sd = variant_seed_dirs(variant, seeds=seeds, project_root=project_root)
    if not sd:
        return None
    rows = []
    for rdir in sd.values():
        rd = ah.read_csv(rdir / "rounds.csv")
        if rd is None or rd.empty:
            continue
        cols = [f"AP_{c}" for c in classes if f"AP_{c}" in rd.columns]
        if not cols:
            continue
        tail = rd.tail(tail_k)[cols]
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
    """Wide ``class x variant`` grid of smoothed end-of-training per-class AP."""
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
    """Long-format per-class AP trajectory through the rounds (mean over seeds).

    Columns: ``round, items_processed_total, optimizer_steps_total, class, AP``.
    """
    sd = variant_seed_dirs(variant, seeds=seeds, project_root=project_root)
    parts = []
    for rdir in sd.values():
        rd = ah.read_csv(rdir / "rounds.csv")
        if rd is None or rd.empty:
            continue
        keep = [c for c in
                ("round", "items_processed_total", "optimizer_steps_total")
                if c in rd.columns]
        ap_cols = [f"AP_{c}" for c in classes if f"AP_{c}" in rd.columns]
        if not ap_cols:
            continue
        long = rd[keep + ap_cols].melt(id_vars=keep, var_name="ap_col", value_name="AP")
        long["class"] = long["ap_col"].str.replace("AP_", "", regex=False)
        parts.append(long.drop(columns=["ap_col"]))
    if not parts:
        return pd.DataFrame()
    cat = pd.concat(parts, ignore_index=True)
    group_cols = [c for c in
                  ("round", "items_processed_total", "optimizer_steps_total", "class")
                  if c in cat.columns]
    return cat.groupby(group_cols, as_index=False)["AP"].mean()


# =============================================================================
# Refresh-segment accept rate (within-stream decay diagnostic, adaptive only)
# =============================================================================

def refresh_segment_table(
    variants: Sequence[str],
    *,
    project_root: Optional[Path] = None,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Accept rate inside each inter-refresh segment for adaptive variants.

    Reads `decisions.csv` and `refreshes.csv` for one seed (default 42)
    per variant.  Returns columns ``variant, label, refresh_idx,
    segment_start, segment_end, n_frames, n_accepts, accept_rate``.
    """
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
# Convenience: build the full set of federated summary tables in one call
# =============================================================================

def build_summary_tables(
    *,
    project_root: Optional[Path] = None,
    tail_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Compute every standard summary table; returns a dict keyed by table name.

    Tables emitted:

    - ``inventory``
    - ``iso_accept``
    - ``ablation_twoRef``
    - ``ablation_tighter_accept``
    - ``ablation_heavyLocal``
    - ``ablation_static_vs_adaptive``
    - ``per_client_accept``
    - ``novelty_routing``
    - ``per_domain_curated``      (block x variant grid, smoothed tail-k mAP)
    - ``per_domain_summary_curated``
    - ``per_domain_delta``         (block x pairing grid)
    - ``per_domain_delta_summary``
    - ``per_block_trajectory_delta``
    - ``per_class_curated``
    """
    project_root = project_root or ah.find_project_root()
    prime_registry(project_root=project_root)

    inv = inventory_table(project_root=project_root, tail_k=tail_k)

    cur_variants = [v for v in FEATURED_VARIANTS
                    if manifest_for_variant(v) == "curated"
                    and schedule_for_variant(v) == "default"]
    hl_variants = [v for v in FEATURED_VARIANTS
                   if schedule_for_variant(v) == "heavyLocal"]

    return {
        "inventory": inv,
        "iso_accept": iso_accept_table(project_root=project_root, tail_k=tail_k),
        "ablation_twoRef": ablation_pair_table(
            ABLATION_PAIRINGS["twoRef"], project_root=project_root, tail_k=tail_k),
        "ablation_tighter_accept": ablation_pair_table(
            ABLATION_PAIRINGS["tighter_accept"], project_root=project_root, tail_k=tail_k),
        "ablation_heavyLocal": ablation_pair_table(
            ABLATION_PAIRINGS["heavyLocal"], project_root=project_root, tail_k=tail_k),
        "ablation_static_vs_adaptive": ablation_pair_table(
            ABLATION_PAIRINGS["static_vs_adaptive"],
            project_root=project_root, tail_k=tail_k),
        "per_client_accept": per_client_accept_table(
            project_root=project_root),
        "novelty_routing": novelty_routing_summary(
            project_root=project_root),
        "per_domain_curated": per_domain_grid(
            cur_variants, project_root=project_root, tail_k=tail_k),
        "per_domain_summary_curated": per_domain_summary(
            cur_variants, project_root=project_root, tail_k=tail_k),
        "per_domain_heavyLocal": per_domain_grid(
            hl_variants, project_root=project_root, tail_k=tail_k),
        "per_domain_delta": per_domain_delta_grid(
            project_root=project_root, tail_k=tail_k),
        "per_domain_delta_summary": per_domain_delta_summary(
            project_root=project_root, tail_k=tail_k),
        "per_block_trajectory_delta": per_block_trajectory_delta(
            project_root=project_root),
        "per_class_curated": per_class_grid(
            cur_variants, project_root=project_root, tail_k=tail_k),
    }
