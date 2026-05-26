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
        CLIENT_LABEL                   -- 0..3 -> "C<i>" (short id)
        DOMAIN_BLOCK_FAMILY            -- block -> "familiar" | "novel"
        curated_client_boundaries(manifest)  -- stream start-x per curated client
        temporal_client_boundaries(manifest) -- stream start-x per temporal client
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

# Ordered list of variants featured in the federated write-up.  Grouped
# by the chapter's 2 x 2 design: schedule (default vs heavy-local) crossed
# with partition (curated domain-aligned vs temporal time-aligned).  The
# order is also the row order for tables and the legend order for
# figures.
FEATURED_VARIANTS: List[str] = [
    # ----- Default schedule, curated partition (headline cell) -----
    # Reference baselines.
    "fed_no_filter_cityday_curated",
    "fed_static_p20_cityday_curated",
    # Filter grid: window / reservoir x single-ref / two-ref x p10..p20.
    "fed_adaptive_window_p10_cityday_curated",
    "fed_adaptive_window_p20_cityday_curated",
    "fed_adaptive_window_p20_twoRef_cityday_curated",
    "fed_adaptive_reservoir_p10_cityday_curated",
    "fed_adaptive_reservoir_p10_twoRef_cityday_curated",
    "fed_adaptive_reservoir_p20_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    # Iso-accept random partners (matched to the empirical filter accepts).
    "fed_random_p7_cityday_curated",
    "fed_random_p12_cityday_curated",
    "fed_random_p15_cityday_curated",
    "fed_random_p18_cityday_curated",
    "fed_random_p77_cityday_curated",

    # ----- Default schedule, temporal partition -----
    "fed_no_filter_cityday_temporal",
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal",
    "fed_random_p15_cityday_temporal",
    "fed_random_p19_cityday_temporal",

    # ----- Heavy-local schedule, curated partition (stress-test cell) -----
    # 10 rounds x 3000 items per client (vs default 30 x 1000) -- same
    # total budget, 3x fewer aggregations, 3x heavier per-round local work.
    "fed_no_filter_cityday_curated_heavyLocal",
    "fed_adaptive_window_p20_cityday_curated_heavyLocal",
    "fed_adaptive_window_p20_twoRef_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p10_twoRef_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p15_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p15_twoRef_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
    "fed_random_p16_cityday_curated_heavyLocal",
    "fed_random_p21_cityday_curated_heavyLocal",
    "fed_random_p26_cityday_curated_heavyLocal",

    # ----- Heavy-local schedule, temporal partition (recovery cell) -----
    "fed_no_filter_cityday_temporal_heavyLocal",
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal_heavyLocal",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal_heavyLocal",
    "fed_random_p25_cityday_temporal_heavyLocal",
    "fed_random_p30_cityday_temporal_heavyLocal",
]

# Short display labels.  Missing entries fall back to the variant name.
# "(HL)" tags the heavy-local schedule; "(T)" tags the temporal manifest.
VARIANT_LABEL: Dict[str, str] = {
    # ----- Default + curated -----
    "fed_no_filter_cityday_curated":                              "none",
    "fed_static_p20_cityday_curated":                             "static p20",
    "fed_adaptive_window_p10_cityday_curated":                    "window p10",
    "fed_adaptive_window_p20_cityday_curated":                    "window p20",
    "fed_adaptive_window_p20_twoRef_cityday_curated":             "window p20 twoRef",
    "fed_adaptive_reservoir_p10_cityday_curated":                 "reservoir p10",
    "fed_adaptive_reservoir_p10_twoRef_cityday_curated":          "reservoir p10 twoRef",
    "fed_adaptive_reservoir_p20_cityday_curated":                 "reservoir p20",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated":          "reservoir p20 twoRef",
    "fed_random_p7_cityday_curated":                              "random p7",
    "fed_random_p12_cityday_curated":                             "random p12",
    "fed_random_p15_cityday_curated":                             "random p15",
    "fed_random_p18_cityday_curated":                             "random p18",
    "fed_random_p77_cityday_curated":                             "random p77",
    # ----- Default + temporal -----
    "fed_no_filter_cityday_temporal":                             "none (T)",
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal":         "reservoir p15 twoRef (T)",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal":         "reservoir p20 twoRef (T)",
    "fed_random_p15_cityday_temporal":                            "random p15 (T)",
    "fed_random_p19_cityday_temporal":                            "random p19 (T)",
    # ----- Heavy-local + curated -----
    "fed_no_filter_cityday_curated_heavyLocal":                   "none (HL)",
    "fed_adaptive_window_p20_cityday_curated_heavyLocal":         "window p20 (HL)",
    "fed_adaptive_window_p20_twoRef_cityday_curated_heavyLocal":  "window p20 twoRef (HL)",
    "fed_adaptive_reservoir_p10_twoRef_cityday_curated_heavyLocal":
                                                                  "reservoir p10 twoRef (HL)",
    "fed_adaptive_reservoir_p15_cityday_curated_heavyLocal":      "reservoir p15 (HL)",
    "fed_adaptive_reservoir_p15_twoRef_cityday_curated_heavyLocal":
                                                                  "reservoir p15 twoRef (HL)",
    "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal":      "reservoir p20 (HL)",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal":
                                                                  "reservoir p20 twoRef (HL)",
    "fed_random_p16_cityday_curated_heavyLocal":                  "random p16 (HL)",
    "fed_random_p21_cityday_curated_heavyLocal":                  "random p21 (HL)",
    "fed_random_p26_cityday_curated_heavyLocal":                  "random p26 (HL)",
    # ----- Heavy-local + temporal -----
    "fed_no_filter_cityday_temporal_heavyLocal":                  "none (T, HL)",
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal_heavyLocal":
                                                                  "reservoir p15 twoRef (T, HL)",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal_heavyLocal":
                                                                  "reservoir p20 twoRef (T, HL)",
    "fed_random_p25_cityday_temporal_heavyLocal":                 "random p25 (T, HL)",
    "fed_random_p30_cityday_temporal_heavyLocal":                 "random p30 (T, HL)",
}


# Print-friendly variant labels (used by ``label_for``).  ``(HL)``
# becomes "(heavy-local)"; ``(T)`` becomes "(temporal)"; ``twoRef`` is
# spelled out as "two-ref"; the threshold percentile is rendered as
# ``\rho`` when it is the random baseline's accept fraction and as
# ``\tau`` when it is the filter's threshold percentile.
THESIS_LABEL: Dict[str, str] = {
    "fed_no_filter_cityday_curated":                              "No filter",
    "fed_static_p20_cityday_curated":                             r"Static ($\tau_{20}$)",
    "fed_adaptive_window_p10_cityday_curated":                    "Window single-ref ($\\tau_{10}$)",
    "fed_adaptive_window_p20_cityday_curated":                    "Window single-ref",
    "fed_adaptive_window_p20_twoRef_cityday_curated":             "Window two-ref",
    "fed_adaptive_reservoir_p10_cityday_curated":                 "Reservoir single-ref ($\\tau_{10}$)",
    "fed_adaptive_reservoir_p10_twoRef_cityday_curated":          "Reservoir two-ref ($\\tau_{10}$)",
    "fed_adaptive_reservoir_p20_cityday_curated":                 "Reservoir single-ref",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated":          "Reservoir two-ref",
    "fed_random_p7_cityday_curated":                              r"Random ($\rho{=}0.07$)",
    "fed_random_p12_cityday_curated":                             r"Random ($\rho{=}0.12$)",
    "fed_random_p15_cityday_curated":                             r"Random ($\rho{=}0.15$)",
    "fed_random_p18_cityday_curated":                             r"Random ($\rho{=}0.18$)",
    "fed_random_p77_cityday_curated":                             r"Random ($\rho{=}0.77$)",
    "fed_no_filter_cityday_temporal":                             "No filter (temporal)",
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal":         "Reservoir two-ref (temporal, $\\tau_{15}$)",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal":         "Reservoir two-ref (temporal)",
    "fed_random_p15_cityday_temporal":                            r"Random ($\rho{=}0.15$, temporal)",
    "fed_random_p19_cityday_temporal":                            r"Random ($\rho{=}0.19$, temporal)",
    "fed_no_filter_cityday_curated_heavyLocal":                   "No filter (heavy-local)",
    "fed_adaptive_window_p20_cityday_curated_heavyLocal":         "Window single-ref (heavy-local)",
    "fed_adaptive_window_p20_twoRef_cityday_curated_heavyLocal":  "Window two-ref (heavy-local)",
    "fed_adaptive_reservoir_p10_twoRef_cityday_curated_heavyLocal":
                                                                  "Reservoir two-ref (heavy-local, $\\tau_{10}$)",
    "fed_adaptive_reservoir_p15_cityday_curated_heavyLocal":      "Reservoir single-ref (heavy-local, $\\tau_{15}$)",
    "fed_adaptive_reservoir_p15_twoRef_cityday_curated_heavyLocal":
                                                                  "Reservoir two-ref (heavy-local, $\\tau_{15}$)",
    "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal":      "Reservoir single-ref (heavy-local)",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal":
                                                                  "Reservoir two-ref (heavy-local)",
    "fed_random_p16_cityday_curated_heavyLocal":                  r"Random ($\rho{=}0.16$, heavy-local)",
    "fed_random_p21_cityday_curated_heavyLocal":                  r"Random ($\rho{=}0.21$, heavy-local)",
    "fed_random_p26_cityday_curated_heavyLocal":                  r"Random ($\rho{=}0.26$, heavy-local)",
    "fed_no_filter_cityday_temporal_heavyLocal":                  "No filter (temporal, heavy-local)",
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal_heavyLocal":
                                                                  "Reservoir two-ref (temporal, heavy-local, $\\tau_{15}$)",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal_heavyLocal":
                                                                  "Reservoir two-ref (temporal, heavy-local)",
    "fed_random_p25_cityday_temporal_heavyLocal":                 r"Random ($\rho{=}0.25$, temporal, heavy-local)",
    "fed_random_p30_cityday_temporal_heavyLocal":                 r"Random ($\rho{=}0.30$, temporal, heavy-local)",
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
    """Return ``"curated"``, ``"temporal"``, ... from the variant name."""
    if "cityday_temporal" in variant:
        return "temporal"
    if "cityday_curated" in variant:
        return "curated"
    return "unknown"


def schedule_for_variant(variant: str) -> str:
    """Return ``"heavyLocal"`` for heavy-local-schedule variants, else ``"default"``."""
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

# Iso-accept pairings: ``(filter_variant, random_variant)`` with the
# random partner's accept_fraction picked to match the filter's empirical
# accept rate (gap typically < 0.01).  Grouped by the chapter's 2 x 2
# design (default / heavy-local x curated / temporal).
ISO_ACCEPT_PAIRINGS: List[Tuple[str, str]] = [
    # ----- Default schedule, curated partition -----
    ("fed_static_p20_cityday_curated",                             "fed_random_p77_cityday_curated"),
    ("fed_adaptive_window_p10_cityday_curated",                    "fed_random_p7_cityday_curated"),
    ("fed_adaptive_window_p20_cityday_curated",                    "fed_random_p12_cityday_curated"),
    ("fed_adaptive_window_p20_twoRef_cityday_curated",             "fed_random_p12_cityday_curated"),
    ("fed_adaptive_reservoir_p10_cityday_curated",                 "fed_random_p7_cityday_curated"),
    ("fed_adaptive_reservoir_p10_twoRef_cityday_curated",          "fed_random_p7_cityday_curated"),
    ("fed_adaptive_reservoir_p20_cityday_curated",                 "fed_random_p18_cityday_curated"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_curated",          "fed_random_p15_cityday_curated"),

    # ----- Default schedule, temporal partition -----
    ("fed_adaptive_reservoir_p15_twoRef_cityday_temporal",         "fed_random_p15_cityday_temporal"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_temporal",         "fed_random_p19_cityday_temporal"),

    # ----- Heavy-local schedule, curated partition -----
    ("fed_adaptive_window_p20_cityday_curated_heavyLocal",
     "fed_random_p26_cityday_curated_heavyLocal"),
    ("fed_adaptive_window_p20_twoRef_cityday_curated_heavyLocal",
     "fed_random_p26_cityday_curated_heavyLocal"),
    ("fed_adaptive_reservoir_p10_twoRef_cityday_curated_heavyLocal",
     "fed_random_p16_cityday_curated_heavyLocal"),
    ("fed_adaptive_reservoir_p15_cityday_curated_heavyLocal",
     "fed_random_p21_cityday_curated_heavyLocal"),
    ("fed_adaptive_reservoir_p15_twoRef_cityday_curated_heavyLocal",
     "fed_random_p21_cityday_curated_heavyLocal"),
    ("fed_adaptive_reservoir_p20_cityday_curated_heavyLocal",
     "fed_random_p26_cityday_curated_heavyLocal"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
     "fed_random_p26_cityday_curated_heavyLocal"),

    # ----- Heavy-local schedule, temporal partition -----
    ("fed_adaptive_reservoir_p15_twoRef_cityday_temporal_heavyLocal",
     "fed_random_p25_cityday_temporal_heavyLocal"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_temporal_heavyLocal",
     "fed_random_p30_cityday_temporal_heavyLocal"),
]

# Ablation pairings: ``(label, baseline_variant, ablated_variant)``.
# delta_smoothed in `ablation_pair_table` reads as
# ``ablated mAP - baseline mAP`` (i.e. positive => the ablation helps).
ABLATION_PAIRINGS: Dict[str, List[Tuple[str, str, str]]] = {
    # Adding the bootstrap Gaussian alongside the adaptive Gaussian
    # (two-reference Mahalanobis).  Stabilises the threshold against
    # reference-set drift; helps window more than reservoir.
    "twoRef": [
        ("Window",
         "fed_adaptive_window_p20_cityday_curated",
         "fed_adaptive_window_p20_twoRef_cityday_curated"),
        ("Reservoir",
         "fed_adaptive_reservoir_p20_cityday_curated",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated"),
        (r"Reservoir ($\tau_{10}$)",
         "fed_adaptive_reservoir_p10_cityday_curated",
         "fed_adaptive_reservoir_p10_twoRef_cityday_curated"),
        ("Window\n(heavy-local)",
         "fed_adaptive_window_p20_cityday_curated_heavyLocal",
         "fed_adaptive_window_p20_twoRef_cityday_curated_heavyLocal"),
        ("Reservoir\n(heavy-local)",
         "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal"),
        (r"Reservoir ($\tau_{15}$)" + "\n(heavy-local)",
         "fed_adaptive_reservoir_p15_cityday_curated_heavyLocal",
         "fed_adaptive_reservoir_p15_twoRef_cityday_curated_heavyLocal"),
    ],
    # Tighter accept (lower threshold percentile).  Tests whether the
    # filter does better with a smaller, more selective label budget.
    "tighter_accept": [
        (r"Window: $\tau_{20}{\to}\tau_{10}$",
         "fed_adaptive_window_p20_cityday_curated",
         "fed_adaptive_window_p10_cityday_curated"),
        (r"Reservoir: $\tau_{20}{\to}\tau_{10}$",
         "fed_adaptive_reservoir_p20_cityday_curated",
         "fed_adaptive_reservoir_p10_cityday_curated"),
        (r"Reservoir two-ref: $\tau_{20}{\to}\tau_{10}$",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
         "fed_adaptive_reservoir_p10_twoRef_cityday_curated"),
        (r"Reservoir: $\tau_{20}{\to}\tau_{15}$" + "\n(heavy-local)",
         "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal",
         "fed_adaptive_reservoir_p15_cityday_curated_heavyLocal"),
        (r"Reservoir two-ref: $\tau_{20}{\to}\tau_{15}$" + "\n(heavy-local)",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
         "fed_adaptive_reservoir_p15_twoRef_cityday_curated_heavyLocal"),
        (r"Reservoir two-ref: $\tau_{20}{\to}\tau_{10}$" + "\n(heavy-local)",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
         "fed_adaptive_reservoir_p10_twoRef_cityday_curated_heavyLocal"),
    ],
    # Heavy-local schedule: 10 rounds x 3000 items (vs default 30 x 1000).
    # Same total budget, 3x fewer aggregations, 3x heavier per-round
    # local work.  Holding accept budget constant within each pair.
    "heavyLocal": [
        ("No filter",
         "fed_no_filter_cityday_curated",
         "fed_no_filter_cityday_curated_heavyLocal"),
        ("Window",
         "fed_adaptive_window_p20_cityday_curated",
         "fed_adaptive_window_p20_cityday_curated_heavyLocal"),
        ("Window\ntwo-ref",
         "fed_adaptive_window_p20_twoRef_cityday_curated",
         "fed_adaptive_window_p20_twoRef_cityday_curated_heavyLocal"),
        ("Reservoir",
         "fed_adaptive_reservoir_p20_cityday_curated",
         "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal"),
        ("Reservoir\ntwo-ref",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal"),
        (r"Reservoir ($\tau_{10}$)" + "\ntwo-ref",
         "fed_adaptive_reservoir_p10_twoRef_cityday_curated",
         "fed_adaptive_reservoir_p10_twoRef_cityday_curated_heavyLocal"),
    ],
    # Adaptive (refreshed reference) vs static (bootstrap-only) on the
    # headline curated cell.
    "static_vs_adaptive": [
        ("Reservoir two-ref vs static",
         "fed_static_p20_cityday_curated",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated"),
    ],
    # Manifest replication: does the headline finding (default + curated)
    # reproduce on the temporal stream order?
    "temporal_replication": [
        ("Reservoir two-ref",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
         "fed_adaptive_reservoir_p20_twoRef_cityday_temporal"),
        ("Reservoir two-ref\n(heavy-local)",
         "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
         "fed_adaptive_reservoir_p20_twoRef_cityday_temporal_heavyLocal"),
    ],
}


# =============================================================================
# Federated client / domain bookkeeping
# =============================================================================

# Display labels for the four `domain_aligned` clients we use.  These
# descriptions match the `domain_client_groups` in the curated configs:
# C0 owns the bootstrap distribution (city_day_clear/cloudy); C1 owns
# the off-bootstrap city blocks (rain_wet, snow, twilight, night); C2
# owns the urban arterial roads; C3 owns highway and rural roads.  The
# old "city_day_novel" label was misleading because it conflated city
# day variants with city night/twilight; the corrected labels make the
# off-time-of-day vs off-weather composition of C1 explicit.
CLIENT_LABEL: Dict[int, str] = {
    0: "C0",
    1: "C1",
    2: "C2",
    3: "C3",
}

# Display labels for the four `contiguous` clients used on the temporal
# manifest.  The contiguous strategy gives each client one chronological
# quartile of the post-bootstrap stream.  Kept short so they fit in a
# single-row plot legend.
TEMPORAL_CLIENT_LABEL: Dict[int, str] = {
    0: "Q1",
    1: "Q2",
    2: "Q3",
    3: "Q4",
}


def client_label(variant: str, client: int) -> str:
    """Pick the right per-client label dict based on the variant's manifest."""
    if "_temporal" in variant:
        return TEMPORAL_CLIENT_LABEL.get(client, f"C{client}")
    return CLIENT_LABEL.get(client, f"C{client}")


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
    # Per-category buckets (time_of_day / road_condition dimensions).
    # Treated as novel for the per-category trajectory figure, where
    # "(novel)" is just appended to the panel title.
    "night":                      "novel",
    "twilight":                   "novel",
    "wet":                        "novel",
    "snow":                       "novel",
    "rain_wet":                   "novel",
}


def block_family(block: str) -> str:
    """Return ``"familiar"``, ``"novel"`` or ``"unknown"`` for a block name."""
    return DOMAIN_BLOCK_FAMILY.get(block, "unknown")


# Coarse client-affinity grouping for the curated `domain_aligned`
# partition.  Each block belongs to exactly one of the four client
# groups; the temporal manifest mixes them across all four clients.
# This is used by the per-client domain-composition figure (the
# four colors map back to the C0/C1/C2/C3 client palette).
CLIENT_GROUP_OF_BLOCK: Dict[str, str] = {
    "city_day_clear":           "city day",
    "city_day_cloudy":          "city day",
    "city_day_rain_wet":        "city night/twi/wet",
    "city_day_snow":            "city night/twi/wet",
    "city_twilight":            "city night/twi/wet",
    "city_night":               "city night/twi/wet",
    "arterial-urban_day":       "urban arterial",
    "arterial-urban_twi-night": "urban arterial",
    "highway_day":              "highway + rural",
    "highway_twi-night":        "highway + rural",
    "arterial-rural_day":       "highway + rural",
    "arterial-rural_twi-night": "highway + rural",
    "smaller-rural_all":        "highway + rural",
}

# Ordered list of the client-affinity groups (used as a stable color /
# legend ordering for the composition stack).
CLIENT_GROUP_ORDER: List[str] = [
    "city day",
    "city night/twi/wet",
    "urban arterial",
    "highway + rural",
]


def client_group_for_block(block: str) -> str:
    """Coarse 4-way grouping of a stream block (matches the curated partition).

    ``"city day"`` is the bootstrap distribution; ``"city night/twi/wet"``
    is the off-bootstrap city blocks; the other two are the road-type
    groupings.  Blocks not in `CLIENT_GROUP_OF_BLOCK` fall back to
    ``"unknown"``.
    """
    return CLIENT_GROUP_OF_BLOCK.get(block, "unknown")


def _stream_total_frames(
    manifest: Optional[Mapping],
    *,
    bootstrap_frames: int = 0,
) -> int:
    """Number of post-bootstrap stream frames implied by a manifest.

    Prefers ``manifest.ordering.block_sizes`` (works for curated
    manifests) and falls back to counting ``split == "train"`` frames
    minus ``bootstrap_frames`` (works for temporal manifests that omit
    block sizes).
    """
    if not manifest:
        return 0
    ordering = manifest.get("ordering") or {}
    block_sizes = dict(ordering.get("block_sizes") or {})
    total = sum(int(v) for v in block_sizes.values())
    if total > 0:
        return total
    frames = manifest.get("frames") or []
    train = sum(1 for f in frames if (f or {}).get("split") == "train")
    return max(0, train - int(bootstrap_frames))


def curated_client_boundaries(
    manifest: Optional[Mapping],
    *,
    bootstrap_frames: int = 0,
) -> List[Tuple[int, str]]:
    """Post-bootstrap frame positions where each curated client starts.

    Walks the manifest block order and records the position where the
    client group changes, giving the stream-coordinate start of each
    domain-aligned client's data.

    Args:
        manifest: Parsed manifest dict from `runs.load_manifest`.
        bootstrap_frames: Unused; retained for API symmetry with
            `temporal_client_boundaries`.

    Returns:
        List of ``(start_frame_idx, label)`` sorted by frame index.
        The first entry is always ``(0, "C0")``.  Returns an empty
        list if the manifest lacks ordering info.
    """
    if not manifest:
        return []
    ordering = manifest.get("ordering") or {}
    block_order = list(ordering.get("block_order") or [])
    block_sizes = dict(ordering.get("block_sizes") or {})
    if not block_order or not block_sizes:
        return []
    boundaries: List[Tuple[int, str]] = []
    pos = 0
    prev_group: Optional[str] = None
    client_idx = 0
    for b in block_order:
        size = int(block_sizes.get(b, 0))
        group = CLIENT_GROUP_OF_BLOCK.get(b)
        if group is not None and group != prev_group:
            boundaries.append((pos, f"C{client_idx}"))
            prev_group = group
            client_idx += 1
        pos += size
    return boundaries


def temporal_client_boundaries(
    manifest: Optional[Mapping],
    *,
    bootstrap_frames: int = 0,
    num_clients: int = 4,
) -> List[Tuple[int, str]]:
    """Post-bootstrap frame positions where each temporal client starts.

    The temporal partition splits the post-bootstrap stream into
    ``num_clients`` equal-sized contiguous slices.

    Args:
        manifest: Parsed manifest dict from `runs.load_manifest`.
        bootstrap_frames: Subtracted from the train-frame count when
            inferring the post-bootstrap stream size from the manifest
            frames (used as a fallback when ``ordering.block_sizes`` is
            missing).
        num_clients: Number of clients (quartiles); defaults to 4.

    Returns:
        List of ``(start_frame_idx, label)`` sorted by frame index.
        The first entry is always ``(0, "Q1")``.  Returns an empty
        list if the manifest provides neither block sizes nor train
        frames.
    """
    if not manifest or num_clients <= 0:
        return []
    total = _stream_total_frames(manifest, bootstrap_frames=bootstrap_frames)
    if total == 0:
        return []
    # Mirror `core.partitioning.partition_frames` (contiguous strategy)
    # inline to avoid pulling the torch-dependent `core` package into
    # the analysis stack.
    base, rem = divmod(total, num_clients)
    out: List[Tuple[int, str]] = []
    offset = 0
    for i in range(num_clients):
        out.append((offset, f"Q{i + 1}"))
        offset += base + (1 if i < rem else 0)
    return out


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
    """Side-by-side stats for each (label, baseline, ablated) triple.

    Columns include both means and the seed std for ``smoothed``,
    so the bar plotter can draw error bars without rejoining tables.
    """
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
            "baseline_smoothed_std": b["smoothed_std"],
            "ablated_smoothed": a["smoothed"],
            "ablated_smoothed_std": a["smoothed_std"],
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


def per_round_per_client_accept(
    variants: Sequence[str],
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Per-(variant, round, client) accept rate, mean across seeds.

    Reads each run's ``rounds.csv`` and computes
    ``client_<i>_accepted / client_<i>_items`` per round.  Used by the
    per-round per-client dynamics figure (the federated analogue of the
    streaming accept-rate-over-time top panel).

    Returns columns:
        ``variant, label, family, schedule, round, client,
        client_label, items, accepted, accept_rate, accept_rate_std,
        n_seeds``.
    """
    rows: List[Dict] = []
    for v in variants:
        sd = variant_seed_dirs(v, seeds=seeds, project_root=project_root)
        if not sd:
            continue
        # Stack per-seed rounds.csv frames keyed by `round` so we can
        # take cross-seed mean / std per (round, client).
        per_seed: List[pd.DataFrame] = []
        for rdir in sd.values():
            rd = ah.read_csv(rdir / "rounds.csv")
            if rd is None or rd.empty:
                continue
            per_seed.append(rd)
        if not per_seed:
            continue
        cat = pd.concat(per_seed, ignore_index=True)
        item_cols = [c for c in cat.columns
                     if c.startswith("client_") and c.endswith("_items")]
        client_ids = sorted(int(c.split("_", 2)[1]) for c in item_cols)
        family = family_for_variant(v, project_root=project_root)
        schedule = schedule_for_variant(v)
        # Compute per-row per-client accept rate (NaN when items==0).
        for cid in client_ids:
            items = cat[f"client_{cid}_items"].astype(float)
            acc = cat[f"client_{cid}_accepted"].astype(float)
            rate = np.where(items > 0, acc / items, np.nan)
            tmp = pd.DataFrame({
                "round": cat["round"],
                "items": items,
                "accepted": acc,
                "rate": rate,
            })
            agg = tmp.groupby("round", as_index=False).agg(
                items=("items", "mean"),
                accepted=("accepted", "mean"),
                accept_rate=("rate", "mean"),
                accept_rate_std=("rate", "std"),
                n_seeds=("rate", "count"),
            )
            agg["variant"] = v
            agg["label"] = label_for(v)
            agg["family"] = family
            agg["schedule"] = schedule
            agg["client"] = cid
            agg["client_label"] = client_label(v, cid)
            rows.append(agg)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[[
        "variant", "label", "family", "schedule",
        "round", "client", "client_label",
        "items", "accepted", "accept_rate", "accept_rate_std", "n_seeds",
    ]]


def per_client_block_composition(
    variant: str,
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Per-client distribution over `stream_block`s for a single variant.

    Joins the run's ``decisions.csv`` (which records the client each
    frame was sent to) with the manifest (which gives every frame's
    ``scene_bucket``).  The resulting fractions describe the *static*
    partition: which blocks each client owns and in what proportion.
    For the curated `domain_aligned` partition this just confirms the
    config; for `contiguous` (temporal) it surfaces the empirical mix
    inside each chronological quartile.

    The first available seed is used because the partition is
    deterministic given a (manifest, partition_strategy) pair.

    Returns columns:
        ``variant, manifest, partition, client, client_label, block,
        n_frames, fraction, group``.
    """
    proj = project_root or ah.find_project_root()
    rdir = latest_seed_dir(variant, project_root=proj)
    if rdir is None:
        # Fall back to whatever seed is available.
        sd = variant_seed_dirs(variant, seeds=seeds, project_root=proj)
        if not sd:
            return pd.DataFrame()
        rdir = next(iter(sd.values()))

    cfg = ah.load_run_config(rdir) or {}
    man = ah.load_manifest(proj, str(cfg.get("manifest_path", "")))
    if man is None:
        return pd.DataFrame()
    frames_df = ah.manifest_to_dataframe(man)
    if frames_df.empty or "scene_bucket" not in frames_df.columns:
        return pd.DataFrame()

    dec = ah.read_csv(rdir / "decisions.csv")
    if dec is None or dec.empty or "client_id" not in dec.columns:
        return pd.DataFrame()
    # Normalise frame_id padding so the merge succeeds.
    dec = dec.copy()
    dec["frame_id"] = dec["frame_id"].astype(str).str.zfill(6)
    fr = frames_df[["frame_id", "scene_bucket"]].copy()
    fr["frame_id"] = fr["frame_id"].astype(str).str.zfill(6)
    merged = dec.merge(fr, on="frame_id", how="left")
    merged = merged.dropna(subset=["scene_bucket"])

    counts = (merged.groupby(["client_id", "scene_bucket"], as_index=False)
              .size().rename(columns={"size": "n_frames",
                                      "scene_bucket": "block",
                                      "client_id": "client"}))
    totals = counts.groupby("client")["n_frames"].transform("sum")
    counts["fraction"] = counts["n_frames"] / totals.replace(0, np.nan)
    counts["client_label"] = counts["client"].map(
        lambda c: client_label(variant, int(c)))
    counts["group"] = counts["block"].map(client_group_for_block)
    counts["variant"] = variant
    counts["manifest"] = manifest_for_variant(variant)
    counts["partition"] = (cfg.get("federated", {}) or {}).get(
        "partition_strategy", "unknown")
    return counts.sort_values(["client", "block"]).reset_index(drop=True)


def _weather_bucket(scraped_weather: Optional[str], road_condition: Optional[str]) -> str:
    """Coarse weather bucket (mirrors `evaluation.stream_blocks._weather_bucket`)."""
    w = (scraped_weather or "").lower()
    rc = (road_condition or "").lower()
    if "snow" in w or "snow" in rc:
        return "snow"
    if "rain" in w or "wet" in rc:
        return "rain_wet"
    if "fog" in w or "cloud" in w or "overcast" in w:
        return "cloudy"
    return "clear"


def per_client_dimension_composition(
    variant: str,
    *,
    fields: Sequence[str] = ("time_of_day", "road_type", "weather"),
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> Dict[str, pd.DataFrame]:
    """Per-client fractional composition along TOD / road / weather dimensions.

    Federated analogue of the streaming `stream_composition` helper:
    each client takes the role of a stream-window slice.  For every
    requested ``field`` we return a wide DataFrame whose index is the
    (sorted) set of client ids and whose columns are the field's
    categorical values, holding the *fraction* of that client's
    accepted-or-rejected frames carrying each value.

    The ``"weather"`` field is derived from the manifest's
    ``scraped_weather`` + ``road_condition`` (mirrors the stream-block
    labeling used at training time).  Other field names are looked up
    directly in the manifest frame metadata.

    Args:
        variant: Variant whose decisions.csv defines the per-frame
            client assignment.  The first available seed is used; the
            partition is deterministic given a manifest, so the choice
            of seed does not matter.
        fields: Dimensions to summarise.  Defaults to the same three
            the streaming chapter uses.

    Returns:
        ``{field: wide_df}`` -- one DataFrame per field, indexed by
        client id.  Empty fields (no data, or all-missing) return an
        empty DataFrame.
    """
    proj = project_root or ah.find_project_root()
    rdir = latest_seed_dir(variant, project_root=proj)
    if rdir is None:
        sd = variant_seed_dirs(variant, seeds=seeds, project_root=proj)
        if not sd:
            return {f: pd.DataFrame() for f in fields}
        rdir = next(iter(sd.values()))

    cfg = ah.load_run_config(rdir) or {}
    man = ah.load_manifest(proj, str(cfg.get("manifest_path", "")))
    if man is None:
        return {f: pd.DataFrame() for f in fields}
    frames_df = ah.manifest_to_dataframe(man)
    if frames_df.empty:
        return {f: pd.DataFrame() for f in fields}

    dec = ah.read_csv(rdir / "decisions.csv")
    if dec is None or dec.empty or "client_id" not in dec.columns:
        return {f: pd.DataFrame() for f in fields}
    dec = dec[["client_id", "frame_id"]].copy()
    dec["frame_id"] = dec["frame_id"].astype(str).str.zfill(6)
    fr = frames_df.copy()
    fr["frame_id"] = fr["frame_id"].astype(str).str.zfill(6)
    if "weather" in fields and "weather" not in fr.columns:
        fr["weather"] = [
            _weather_bucket(row.get("scraped_weather"),
                            row.get("road_condition"))
            for _, row in fr.iterrows()
        ]
    merged = dec.merge(fr, on="frame_id", how="left")

    out: Dict[str, pd.DataFrame] = {}
    for field in fields:
        if field not in merged.columns:
            out[field] = pd.DataFrame()
            continue
        sub = merged.dropna(subset=[field])
        if sub.empty:
            out[field] = pd.DataFrame()
            continue
        counts = (sub.groupby("client_id")[field]
                     .value_counts().unstack(fill_value=0))
        frac = counts.div(counts.sum(axis=1).clip(lower=1), axis=0)
        # Pin the column order using the canonical short-name maps
        # so the legend reads in a familiar order across panels.
        if field == "time_of_day":
            order = ["day", "twilight", "night"]
        elif field == "road_type":
            order = ["city", "arterial-urban", "highway",
                     "arterial-rural", "smaller-rural"]
        elif field == "weather":
            order = ["clear", "cloudy", "rain_wet", "snow"]
        else:
            order = sorted(frac.columns)
        for col in order:
            if col not in frac.columns:
                frac[col] = 0.0
        extras = [c for c in frac.columns if c not in order]
        out[field] = frac[order + extras]
        out[field].index.name = "client"
    return out


def per_block_accept_rate_table(
    variants: Sequence[str],
    *,
    project_root: Optional[Path] = None,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Per-(variant, stream_block) mean accept rate across the run.

    Joins each variant's ``decisions.csv`` with the run's manifest
    (``scene_bucket`` field) and computes the fraction of frames
    accepted per block.  Aggregates across seeds with mean / std.

    This is the federated analogue of `streaming.per_block_routing`:
    one number per (filter, block) describing how much compute the
    variant routes to that block.

    Returns columns:
        ``variant, label, family, schedule, block, group, n_frames,
        accept_rate, accept_rate_std, n_seeds``.
    """
    proj = project_root or ah.find_project_root()
    rows: List[Dict] = []
    for v in variants:
        sd = variant_seed_dirs(v, seeds=seeds, project_root=proj)
        if not sd:
            continue
        per_seed: Dict[str, List[float]] = {}
        per_seed_n: Dict[str, List[int]] = {}
        family = family_for_variant(v, project_root=proj)
        schedule = schedule_for_variant(v)
        for rdir in sd.values():
            cfg = ah.load_run_config(rdir) or {}
            man = ah.load_manifest(proj, str(cfg.get("manifest_path", "")))
            if man is None:
                continue
            frames_df = ah.manifest_to_dataframe(man)
            if frames_df.empty or "scene_bucket" not in frames_df.columns:
                continue
            dec = ah.read_csv(rdir / "decisions.csv")
            if dec is None or dec.empty:
                continue
            dec = dec.copy()
            dec["frame_id"] = dec["frame_id"].astype(str).str.zfill(6)
            fr = frames_df[["frame_id", "scene_bucket"]].copy()
            fr["frame_id"] = fr["frame_id"].astype(str).str.zfill(6)
            merged = dec.merge(fr, on="frame_id", how="left")
            merged = merged.dropna(subset=["scene_bucket"])
            merged["accept"] = (merged["action"] == "accept").astype(int)
            agg = merged.groupby("scene_bucket", as_index=False).agg(
                n_frames=("accept", "size"),
                n_accept=("accept", "sum"),
            )
            agg["accept_rate"] = agg["n_accept"] / agg["n_frames"]
            for _, row in agg.iterrows():
                per_seed.setdefault(row["scene_bucket"], []).append(
                    float(row["accept_rate"]))
                per_seed_n.setdefault(row["scene_bucket"], []).append(
                    int(row["n_frames"]))
        for block, rates in per_seed.items():
            rows.append({
                "variant": v,
                "label": label_for(v),
                "family": family,
                "schedule": schedule,
                "block": block,
                "group": client_group_for_block(block),
                "n_frames": int(np.mean(per_seed_n[block])),
                "accept_rate": float(np.mean(rates)),
                "accept_rate_std": float(np.std(rates)) if len(rates) > 1 else float("nan"),
                "n_seeds": len(rates),
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

    Aggregation is by ``round`` only.  ``items_processed_total`` and
    ``optimizer_steps_total`` are averaged across seeds so they remain
    sensible x-axes for trajectory plots (otherwise per-seed
    differences in accept counts would split each round into multiple
    rows and break the per-round mean).
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
        if "round" not in cat.columns:
            continue
        # Aggregate per `round`: mean / std across seeds for `mAP`,
        # and mean for the cumulative-X columns.
        x_cols = [c for c in ("items_processed_total",
                              "optimizer_steps_total") if c in cat.columns]
        agg_dict: Dict[str, Tuple[str, str]] = {
            "mAP": ("mAP", "mean"),
            "mAP_std": ("mAP", "std"),
            "n": ("mAP", "count"),
        }
        for c in x_cols:
            agg_dict[c] = (c, "mean")
        agg = cat.groupby("round", as_index=False).agg(**agg_dict)
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

    Columns: ``checkpoint_idx, items_processed, optimizer_steps,
    bucket, mAP, mAP_std, n``.

    Aggregation is by ``(checkpoint_idx, bucket)`` only; the cumulative
    counters (``items_processed``, ``optimizer_steps``) are averaged
    across seeds.  Aggregating by them as well would split each
    (checkpoint, block) into per-seed rows because different seeds
    accept slightly different sample counts each round.
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
    if "checkpoint_idx" not in cat.columns or "bucket" not in cat.columns:
        return pd.DataFrame()
    x_cols = [c for c in ("items_processed", "optimizer_steps") if c in cat.columns]
    agg_dict: Dict[str, Tuple[str, str]] = {
        "mAP": ("mAP", "mean"),
        "mAP_std": ("mAP", "std"),
        "n": ("mAP", "count"),
    }
    for c in x_cols:
        agg_dict[c] = (c, "mean")
    return cat.groupby(["checkpoint_idx", "bucket"], as_index=False).agg(
        **agg_dict)


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
