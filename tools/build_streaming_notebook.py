"""Generate notebooks/01_streaming_analysis.ipynb from a structured cell list.

Run as ``python tools/build_streaming_notebook.py`` from the project
root.  Add ``--execute`` to also run the notebook end-to-end so cells
acquire their outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

OUT = (Path(__file__).resolve().parents[1]
       / "notebooks" / "01_streaming_analysis.ipynb")


CELLS: List[Tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.lstrip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.lstrip("\n").rstrip() + "\n"))


# =============================================================================
# 1. Title and setup
# =============================================================================

md("""\
# Streaming detection — analysis

Tables and figures for streaming detection.

Outputs:

- Tables  -> `reports/streaming/tables/*.csv`
- Figures -> `reports/streaming/figures/*.{pdf,png}`
""")

md("## 1 Setup")

code("""\
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

%matplotlib inline

# Make the package importable when stream-active-fl has not been installed
# (e.g. when running the notebook directly from a project clone).
_proj = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
if str(_proj / "src") not in sys.path:
    sys.path.insert(0, str(_proj / "src"))

from stream_active_fl.analysis import runs as ah
from stream_active_fl.analysis import streaming as sa
from stream_active_fl.analysis.figures import streaming as sf

ah.setup_notebook_style(dpi=120)

PROJECT_ROOT = ah.find_project_root()
OUTPUTS = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports" / "streaming"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

#: Tail-k for smoothed mAP (matches the federated chapter).
TAIL_K = 5

sa.prime_registry(project_root=PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")
print(f"Variants registered: {len(sa.FEATURED_VARIANTS)}")
""")

md("""\
### 1a Headline variants and pairs
""")

code("""\
HEADLINE_CUR = [
    "no_filter_cityday_curated",
    "random_p21_cityday_curated",
    "random_p33_cityday_curated",
    "static_p20_cityday_curated",
    "adaptive_window_p20_twoRef_m1500_cityday_curated",
    "adaptive_reservoir_p20_twoRef_m1500_cityday_curated",
]

HEADLINE_PAIRS_CUR = [
    ("adaptive_window_p20_m1500_cityday_curated",          "random_p26_cityday_curated"),
    ("adaptive_window_p20_twoRef_m1500_cityday_curated",   "random_p27_cityday_curated"),
    ("adaptive_reservoir_p20_m1500_cityday_curated",       "random_p23_cityday_curated"),
    ("adaptive_reservoir_p20_twoRef_m1500_cityday_curated","random_p21_cityday_curated"),
]

HEADLINE_TMP = [
    "no_filter_cityday_temporal",
    "random_p21_cityday_temporal",
    "random_p28_cityday_temporal",
    "adaptive_window_p20_twoRef_cityday_temporal",
    "adaptive_reservoir_p20_twoRef_cityday_temporal",
]

HEADLINE_PAIRS_TMP = [
    ("adaptive_window_p20_cityday_temporal",          "random_p28_cityday_temporal"),
    ("adaptive_window_p20_twoRef_cityday_temporal",   "random_p31_cityday_temporal"),
    ("adaptive_reservoir_p20_cityday_temporal",       "random_p21_cityday_temporal"),
    ("adaptive_reservoir_p20_twoRef_cityday_temporal","random_p21_cityday_temporal"),
]
""")

# =============================================================================
# 2. Inventory
# =============================================================================

md("""\
## 2 Variant inventory

One row per (variant, manifest) with empirical accept rate, smoothed
tail-`TAIL_K` mAP across 3 seeds, and the final optimizer-step count.""")

code("""\
inv = sa.inventory_table(project_root=PROJECT_ROOT, tail_k=TAIL_K)
inv.to_csv(TABLE_DIR / "inventory.csv", index=False)
display_cols = ["label", "manifest", "family", "n_seeds",
                "accept_rate", "smoothed_mAP", "smoothed_std", "final_optim_steps"]
inv[display_cols].round(4)
""")

md("""\
### 2b Iso-accept leaderboard

Smoothed tail-`TAIL_K` mAP vs effective accept rate, restricted to the
headline variants and the no-filter ceiling.  The dotted grey
curve is the iso-accept random envelope, drawn from all random
baselines on the curated manifest.  Filters above the curve beat
random at the same accept budget.""")

code("""\
LEADERBOARD_HEADLINE = [
    "no_filter_cityday_curated",
    "static_p20_cityday_curated",
    "adaptive_window_p20_m1500_cityday_curated",
    "adaptive_window_p20_twoRef_m1500_cityday_curated",
    "adaptive_reservoir_p20_m1500_cityday_curated",
    "adaptive_reservoir_p20_twoRef_m1500_cityday_curated",
]
fig, _ = sf.plot_inventory_scatter(
    inv, manifest="curated",
    headline_variants=LEADERBOARD_HEADLINE,
    annotate_filters_only=True)
sf.save_figure(fig, "00_iso_accept_leaderboard_curated", out_dir=FIG_DIR)
plt.show()
""")

# Resolve manifest + bootstrap once so accept-dynamics and trajectory
# panels share the same boundaries.
code("""\
sample_dir = sa.latest_seed_dir("no_filter_cityday_curated", 42, project_root=PROJECT_ROOT)
sample_cfg = ah.load_run_config(sample_dir) if sample_dir else {}
MAN_CUR = ah.load_manifest(PROJECT_ROOT, sample_cfg.get("manifest_path") if sample_cfg else None)
BOOT_CUR = ah.get_bootstrap_size(MAN_CUR, sample_cfg)
BOUNDS_CUR, MIDPOINTS_CUR = sa.block_boundaries_and_midpoints(MAN_CUR, bootstrap_frames=BOOT_CUR)

ACCEPT_WINDOW = 500

ROAD_TYPE_ORDER = ("city", "arterial-urban", "highway",
                   "arterial-rural", "smaller-rural")
WEATHER_ORDER = ("clear", "cloudy", "rain_wet", "snow")

def _weather_bucket(frame):
    # Mirrors tools/preprocessing/build_manifests.py::_weather_bucket
    # so the composition panel uses the same 4-bucket grouping the
    # curated manifest is built around (fog folded into cloudy).
    w = (frame.get("scraped_weather") or "").lower()
    rc = (frame.get("road_condition") or "").lower()
    if "snow" in w or "snow" in rc:
        return "snow"
    if "rain" in w or "wet" in rc:
        return "rain_wet"
    if "cloud" in w or "fog" in w or "overcast" in w:
        return "cloudy"
    return "clear"

COMPOSITION_CUR = sa.stream_composition(
    MAN_CUR, bootstrap_frames=BOOT_CUR,
    fields=("time_of_day", "road_type", "weather"),
    window=ACCEPT_WINDOW,
    field_orders={
        "time_of_day": ("day", "twilight", "night"),
        "road_type":   ROAD_TYPE_ORDER,
        "weather":     WEATHER_ORDER,
    },
    field_derivers={"weather": _weather_bucket},
)

TOD_PALETTE = {"day":      ah.TOD_COLORS["day"],
               "twilight": ah.TOD_COLORS["twilight"],
               "night":    ah.TOD_COLORS["night"]}
# Road-type palette: blue family for urban, green for rural, neutral
# tan for highway -- chosen to avoid clashing with the family-color
# palette in the upper accept-rate panel (window=orange, etc.).
ROAD_TYPE_PALETTE = {
    "city":           "#1f78b4",
    "arterial-urban": "#a6cee3",
    "highway":        "#fdbf6f",
    "arterial-rural": "#b2df8a",
    "smaller-rural":  "#33a02c",
}
WEATHER_PALETTE = {
    "clear":    ah.WEATHER_COLORS["clear"],
    "cloudy":   ah.WEATHER_COLORS["cloudy"],
    "rain_wet": ah.WEATHER_COLORS["rain_wet"],
    "snow":     ah.WEATHER_COLORS["snow"],
}
COMP_PALETTES = {
    "time_of_day": TOD_PALETTE,
    "road_type":   ROAD_TYPE_PALETTE,
    "weather":     WEATHER_PALETTE,
}
COMP_TITLES = {
    "time_of_day": "Time-of-day composition of stream window",
    "road_type":   "Road-type composition of stream window",
    "weather":     "Weather composition of stream window",
}
print("Curated stream:", len(BOUNDS_CUR) - 1, "blocks; boundaries =", BOUNDS_CUR)
""")

# =============================================================================
# 3. Accept dynamics
# =============================================================================

md("""\
## 3 Accept dynamics
""")

md("""\
### 3a Static filter — per-category routing
""")

code("""\
STATIC_INTERP_VARIANTS = [
    "static_p20_cityday_curated",
    "random_p21_cityday_curated",
]
INTERP_PANELS = ["time_of_day", "weather"]
static_interp = sa.per_category_routing(
    STATIC_INTERP_VARIANTS, project_root=PROJECT_ROOT)
for cat in INTERP_PANELS:
    if cat in static_interp and not static_interp[cat].empty:
        static_interp[cat].to_csv(
            TABLE_DIR / f"static_per_category_routing_{cat}.csv")
        display(static_interp[cat].round(3))
""")

md("""\
### 3b Static filter — accept rate over the stream
""")

code("""\
STATIC_DYN_VARIANTS = [
    "static_p20_cityday_curated",
]
accept_static = {
    v: sa.windowed_accept_rate_aggregated(v, project_root=PROJECT_ROOT, window=ACCEPT_WINDOW)
    for v in STATIC_DYN_VARIANTS
}
accept_static = {v: df for v, df in accept_static.items() if not df.empty}
fig, _ = sf.plot_rolling_accept_rate(
    accept_static,
    boundaries=BOUNDS_CUR,
    midpoints=MIDPOINTS_CUR,
    composition=COMPOSITION_CUR,
    composition_palettes=COMP_PALETTES,
    composition_titles=COMP_TITLES,
    window=ACCEPT_WINDOW)
sf.save_figure(fig, "02b_accept_rate_with_composition_static_curated", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 3c Per-block accept rate (all filters)
""")

code("""\
ROUTING_VARIANTS = [
    "static_p20_cityday_curated",
    "adaptive_window_p20_m1500_cityday_curated",
    "adaptive_window_p20_twoRef_m1500_cityday_curated",
    "adaptive_reservoir_p20_m1500_cityday_curated",
    "adaptive_reservoir_p20_twoRef_m1500_cityday_curated",
]
routing, routing_std = sa.per_block_routing(
    ROUTING_VARIANTS, project_root=PROJECT_ROOT, return_std=True)
routing.to_csv(TABLE_DIR / "per_block_routing_curated.csv")
routing_std.to_csv(TABLE_DIR / "per_block_routing_curated_std.csv")
display(routing.round(3))
ROUTING_FILTERS_LINES = [
    "adaptive_window_p20_m1500_cityday_curated",
    "adaptive_window_p20_twoRef_m1500_cityday_curated",
    "adaptive_reservoir_p20_m1500_cityday_curated",
    "adaptive_reservoir_p20_twoRef_m1500_cityday_curated",
]
fig, _ = sf.plot_per_block_routing_lines(
    routing,
    filter_variants=ROUTING_FILTERS_LINES,
    random_refs=None,
    static_variant="static_p20_cityday_curated",
    std_grid=routing_std,
    ymax=0.55)
sf.save_figure(fig, "01_per_block_routing_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 3d Accept rate per stream window — adaptive filters")

code("""\
ADAPTIVE_DYN_VARIANTS = [
    "adaptive_window_p20_m1500_cityday_curated",
    "adaptive_window_p20_twoRef_m1500_cityday_curated",
    "adaptive_reservoir_p20_m1500_cityday_curated",
    "adaptive_reservoir_p20_twoRef_m1500_cityday_curated",
]
accept_adapt = {
    v: sa.windowed_accept_rate_aggregated(v, project_root=PROJECT_ROOT, window=ACCEPT_WINDOW)
    for v in ADAPTIVE_DYN_VARIANTS
}
accept_adapt = {v: df for v, df in accept_adapt.items() if not df.empty}
fig, _ = sf.plot_rolling_accept_rate(
    accept_adapt,
    boundaries=BOUNDS_CUR,
    midpoints=MIDPOINTS_CUR,
    composition=COMPOSITION_CUR,
    composition_palettes=COMP_PALETTES,
    composition_titles=COMP_TITLES,
    window=ACCEPT_WINDOW)
sf.save_figure(fig, "02_accept_rate_with_composition_curated", out_dir=FIG_DIR)
plt.show()
""")

# =============================================================================
# 4. Per-domain performance
# =============================================================================

md("""\
## 4 Per-domain performance
""")

md("### 4a Per-domain end-of-stream mAP — headline variants (curated)")

code("""\
grid_cur = sa.per_domain_grid(HEADLINE_CUR, project_root=PROJECT_ROOT, tail_k=TAIL_K)
grid_cur.to_csv(TABLE_DIR / "per_domain_curated.csv")
# Balanced (mean over blocks) + worst-block summary, useful as a
# results-chapter table without exposing the full 12-block grid.
summary_cur = sa.per_domain_summary(
    HEADLINE_CUR, project_root=PROJECT_ROOT, tail_k=TAIL_K)
summary_cur.to_csv(TABLE_DIR / "per_domain_summary_curated.csv", index=False)
display(summary_cur.round(4))
fig, _ = sf.plot_per_domain_heatmap(
    grid_cur)
sf.save_figure(fig, "03_per_domain_heatmap_curated", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 4b Per-block Δ-mAP vs iso-accept random (curated)

Each column is a (filter, iso-accept random) pair from the matched-memory
m=1500 grid.  Positive (red) cells mean the filter beats its random
partner on that block.""")

code("""\
delta_grid = sa.per_domain_delta_grid(
    HEADLINE_PAIRS_CUR, project_root=PROJECT_ROOT, tail_k=TAIL_K)
delta_summary = sa.per_domain_delta_summary(
    HEADLINE_PAIRS_CUR, project_root=PROJECT_ROOT, tail_k=TAIL_K)
delta_grid.to_csv(TABLE_DIR / "per_domain_delta.csv")
delta_summary.to_csv(TABLE_DIR / "per_domain_delta_summary.csv", index=False)
display(delta_summary.round(4))
fig, _ = sf.plot_per_domain_delta_heatmap(
    delta_grid)
sf.save_figure(fig, "04_per_domain_delta_heatmap", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 4c Iso-accept gain — headline scatter

Each marker is one (filter, iso-accept random) pair.  Points above the
zero line beat random at the same accept budget.""")

code("""\
iso_headline = sa.iso_accept_table(
    HEADLINE_PAIRS_CUR, project_root=PROJECT_ROOT, tail_k=TAIL_K)
iso_headline.to_csv(TABLE_DIR / "iso_accept_m1500.csv", index=False)
iso_cols = ["filter_label", "filter_accept", "random_label", "random_accept",
            "accept_gap", "filter_smoothed", "random_smoothed", "delta_smoothed"]
display(iso_headline[iso_cols].round(4))
fig, _ = sf.plot_iso_accept_scatter(iso_headline, manifest="curated")
sf.save_figure(fig, "05_iso_accept_delta_curated", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 4d Trajectory by content category

**Bucketing convention.** Categories are taken from the raw frame
metadata: `night` and `twilight` are the time_of_day field;
`wet` and `snow` are the road_condition field.  The `wet`
panel here is *not* identical to the derived `rain_wet` bucket
shown in the composition figures (which is "scraped_weather
contains rain OR road_condition is wet"); per-domain mAP is only
evaluated against the marginal `road_condition` axis, so the panel
title matches the data column verbatim.  Adding `weather_bucket` as
a fourth evaluation dimension would require re-running all
streaming experiments.""")

code("""\
def _category_trajs(variants):
    parts_tod = {
        v: sa.per_domain_trajectory(
            v, ["night", "twilight"], dim="time_of_day", project_root=PROJECT_ROOT)
        for v in variants
    }
    parts_rc = {
        v: sa.per_domain_trajectory(
            v, ["snow", "wet"], dim="road_condition", project_root=PROJECT_ROOT)
        for v in variants
    }
    out = {}
    for v in variants:
        parts = [df for df in (parts_tod.get(v), parts_rc.get(v))
                 if df is not None and not df.empty]
        if parts:
            out[v] = pd.concat(parts, ignore_index=True)
    return out

CATEGORY_BUCKETS = ["night", "twilight", "wet", "snow"]

# Fig 06: Window family (iso-accept random_p27).  Window two-ref's
# empirical accept rate is 0.272; random_p27 lands at 0.272 (gap 0).
TRAJ_VARIANTS_WIN = [
    "no_filter_cityday_curated",
    "random_p27_cityday_curated",
    "adaptive_window_p20_twoRef_m1500_cityday_curated",
]
fig, _ = sf.plot_per_block_trajectory(
    _category_trajs(TRAJ_VARIANTS_WIN), CATEGORY_BUCKETS,
    x_col="items_processed", n_cols=2, smoothing_window=5,
    active_intervals=None)
sf.save_figure(fig, "06_per_category_trajectory_window", out_dir=FIG_DIR)
plt.show()

# Fig 06b: Reservoir family (iso-accept random_p21).  Reservoir
# two-ref's empirical accept rate is 0.213; random_p21 lands at 0.213
# (gap 0).
TRAJ_VARIANTS_RES = [
    "no_filter_cityday_curated",
    "random_p21_cityday_curated",
    "adaptive_reservoir_p20_twoRef_m1500_cityday_curated",
]
fig, _ = sf.plot_per_block_trajectory(
    _category_trajs(TRAJ_VARIANTS_RES), CATEGORY_BUCKETS,
    x_col="items_processed", n_cols=2, smoothing_window=5,
    active_intervals=None)
sf.save_figure(fig, "06b_per_category_trajectory_reservoir", out_dir=FIG_DIR)
plt.show()
""")

# =============================================================================
# 5. Cross-stream-order replication (temporal manifest)
# =============================================================================

md("""\
## 5 Cross-stream-order replication (temporal manifest)
""")

code("""\
sample_dir_tmp = sa.latest_seed_dir("no_filter_cityday_temporal", 42, project_root=PROJECT_ROOT)
sample_cfg_tmp = ah.load_run_config(sample_dir_tmp) if sample_dir_tmp else {}
MAN_TMP = ah.load_manifest(PROJECT_ROOT, sample_cfg_tmp.get("manifest_path") if sample_cfg_tmp else None)
BOOT_TMP = ah.get_bootstrap_size(MAN_TMP, sample_cfg_tmp)
BOUNDS_TMP, MIDPOINTS_TMP = sa.block_boundaries_and_midpoints(MAN_TMP, bootstrap_frames=BOOT_TMP)
COMPOSITION_TMP = sa.stream_composition(
    MAN_TMP, bootstrap_frames=BOOT_TMP,
    fields=("time_of_day", "road_type", "weather"),
    window=ACCEPT_WINDOW,
    field_orders={
        "time_of_day": ("day", "twilight", "night"),
        "road_type":   ROAD_TYPE_ORDER,
        "weather":     WEATHER_ORDER,
    },
    field_derivers={"weather": _weather_bucket},
)
print(f"Temporal stream: {len(BOUNDS_TMP) - 1} blocks; bootstrap={BOOT_TMP}")
""")

md("### 5a Accept dynamics (temporal)")

code("""\
ACCEPT_DYN_VARIANTS_TMP = [
    "adaptive_window_p20_cityday_temporal",
    "adaptive_window_p20_twoRef_cityday_temporal",
    "adaptive_reservoir_p20_cityday_temporal",
    "adaptive_reservoir_p20_twoRef_cityday_temporal",
]
accept_by_var_tmp = {
    v: sa.windowed_accept_rate_aggregated(v, project_root=PROJECT_ROOT, window=ACCEPT_WINDOW)
    for v in ACCEPT_DYN_VARIANTS_TMP
}
accept_by_var_tmp = {v: df for v, df in accept_by_var_tmp.items() if not df.empty}
fig, _ = sf.plot_rolling_accept_rate(
    accept_by_var_tmp,
    boundaries=BOUNDS_TMP,
    midpoints=MIDPOINTS_TMP,
    composition=COMPOSITION_TMP,
    composition_palettes=COMP_PALETTES,
    composition_titles=COMP_TITLES,
    window=ACCEPT_WINDOW)
sf.save_figure(fig, "07_accept_rate_with_composition_temporal", out_dir=FIG_DIR)
plt.show()
""")

md("### 5b Per-domain mAP and Δ-mAP (temporal)")

code("""\
grid_tmp = sa.per_domain_grid(HEADLINE_TMP, project_root=PROJECT_ROOT, tail_k=TAIL_K)
grid_tmp.to_csv(TABLE_DIR / "per_domain_temporal.csv")
summary_tmp = sa.per_domain_summary(
    HEADLINE_TMP, project_root=PROJECT_ROOT, tail_k=TAIL_K)
summary_tmp.to_csv(TABLE_DIR / "per_domain_summary_temporal.csv", index=False)
display(summary_tmp.round(4))
fig, _ = sf.plot_per_domain_heatmap(
    grid_tmp)
sf.save_figure(fig, "08_per_domain_heatmap_temporal", out_dir=FIG_DIR)
plt.show()

delta_grid_tmp = sa.per_domain_delta_grid(
    HEADLINE_PAIRS_TMP, project_root=PROJECT_ROOT, tail_k=TAIL_K)
fig, _ = sf.plot_per_domain_delta_heatmap(
    delta_grid_tmp)
sf.save_figure(fig, "09_per_domain_delta_heatmap_temporal", out_dir=FIG_DIR)
plt.show()
""")

# =============================================================================
# 6. Methodology ablations
# =============================================================================

md("""\
## 6 Methodology ablations
""")

md("### 6a Static vs adaptive at p20")

code("""\
fig, _ = sf.plot_static_vs_adaptive(inv, manifest="curated")
sf.save_figure(fig, "10_static_vs_adaptive_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 6b Two-reference vs single-reference Mahalanobis")

code("""\
two_ref = sa.ablation_pair_table(
    sa.ABLATION_PAIRINGS["twoRef"], project_root=PROJECT_ROOT, tail_k=TAIL_K)
two_ref.to_csv(TABLE_DIR / "ablation_twoRef.csv", index=False)
display(two_ref.round(4))
fig, _ = sf.plot_ablation_pair_bar(
    two_ref)
sf.save_figure(fig, "11_ablation_twoRef", out_dir=FIG_DIR)
plt.show()
""")

md("### 6c Bootstrap-anchor ablation (noBoot)")

code("""\
no_boot = sa.ablation_pair_table(
    sa.ABLATION_PAIRINGS["noBoot"], project_root=PROJECT_ROOT, tail_k=TAIL_K)
no_boot.to_csv(TABLE_DIR / "ablation_noBoot.csv", index=False)
display(no_boot.round(4))
fig, _ = sf.plot_ablation_pair_bar(
    no_boot)
sf.save_figure(fig, "12_ablation_noBoot", out_dir=FIG_DIR)
plt.show()
""")

md("### 6d Within-refresh accept-rate dynamics")

code("""\
REFRESH_VARIANTS = [
    "adaptive_window_p20_cityday_curated",
    "adaptive_window_p20_twoRef_cityday_curated",
    "adaptive_reservoir_p20_cityday_curated",
    "adaptive_reservoir_p20_twoRef_cityday_curated",
]
seg = sa.refresh_segment_table(REFRESH_VARIANTS, project_root=PROJECT_ROOT, seed=42)
seg.to_csv(TABLE_DIR / "refresh_segments_curated.csv", index=False)
fig, _ = sf.plot_refresh_segment_decay(
    seg)
sf.save_figure(fig, "13_refresh_segment_decay_curated", out_dir=FIG_DIR)
plt.show()
""")

# =============================================================================
# 7. Iso-accept fairness (full table)
# =============================================================================

md("""\
## 7 Iso-accept fairness (full sweep)
""")

code("""\
iso = sa.iso_accept_table(project_root=PROJECT_ROOT, tail_k=TAIL_K)
iso.to_csv(TABLE_DIR / "iso_accept.csv", index=False)
iso_display_cols = ["filter_label", "manifest", "filter_accept", "filter_smoothed",
                    "random_label", "random_accept", "random_smoothed",
                    "accept_gap", "delta_smoothed"]
iso[iso_display_cols].round(4)
""")

# =============================================================================
# 8. Supporting analyses
# =============================================================================

md("""\
## 8 Supporting analyses
""")

md("### 8a Full per-domain heatmap (all featured variants)")

code("""\
cur_variants = [v for v in sa.FEATURED_VARIANTS if sa.manifest_for_variant(v) == "curated"]
grid_full = sa.per_domain_grid(cur_variants, project_root=PROJECT_ROOT, tail_k=TAIL_K)
fig, _ = sf.plot_per_domain_heatmap(
    grid_full)
sf.save_figure(fig, "A0_per_domain_heatmap_curated_full", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 8b Iso-accept leaderboard (full sweep)
""")

code("""\
fig, _ = sf.plot_inventory_scatter(
    inv, manifest="curated",
    annotate_filters_only=False,
    figsize=(8.5, 5.4))
sf.save_figure(fig, "A0b_iso_accept_leaderboard_curated_full", out_dir=FIG_DIR)
plt.show()
""")

md("### 8c Per-class breakdown (headline variants)")

code("""\
PER_CLASS_VARIANTS = HEADLINE_CUR
per_class = sa.per_class_grid(PER_CLASS_VARIANTS, project_root=PROJECT_ROOT, tail_k=TAIL_K)
per_class.to_csv(TABLE_DIR / "per_class_curated.csv")
display(per_class.round(4))
fig, _ = sf.plot_per_class_heatmap(
    per_class)
sf.save_figure(fig, "A1_per_class_heatmap_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 8d Forgetting analysis (early vs late stream)")

code("""\
forget = sa.forgetting_table(
    PER_CLASS_VARIANTS, project_root=PROJECT_ROOT, n_bins=4)
forget.to_csv(TABLE_DIR / "forgetting_curated.csv", index=False)
fig, _ = sf.plot_forgetting_heatmap(
    forget, metric="delta")
sf.save_figure(fig, "A2_forgetting_delta_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 8e Compute efficiency (steps to reach a target mAP)")

code("""\
TARGETS = [0.20, 0.22, 0.24]
EFFICIENCY_VARIANTS = HEADLINE_CUR
steps_table = sa.steps_to_reach_table(
    EFFICIENCY_VARIANTS, TARGETS,
    project_root=PROJECT_ROOT, x_col="optimizer_steps")
steps_table.to_csv(TABLE_DIR / "steps_to_reach.csv", index=False)
display(steps_table.pivot(index="target_mAP", columns="label",
                          values="optimizer_steps").round(0))
fig, _ = sf.plot_steps_to_target(
    steps_table, x_col="optimizer_steps")
sf.save_figure(fig, "A3_steps_to_target_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 8f Overall validation mAP through the stream")

code("""\
TRAJ_HEADLINE = [v for v in HEADLINE_CUR if v in sa.FEATURED_VARIANTS]
traj_cur = sa.mAP_trajectory(TRAJ_HEADLINE, project_root=PROJECT_ROOT)
block_trans = list(zip(BOUNDS_CUR, [m[1] for m in MIDPOINTS_CUR] + [""]))
fig, _ = sf.plot_overall_mAP_trajectory(
    traj_cur, x_col="items_processed", block_transitions=block_trans)
sf.save_figure(fig, "A4_overall_mAP_trajectory_curated", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 9. Index of saved figures
# =============================================================================

md("## 9 Saved figures and tables index")

code("""\
saved = sorted(p for p in FIG_DIR.iterdir() if p.suffix == ".pdf")
print(f"{len(saved)} figures under {FIG_DIR}:")
for p in saved:
    print(f"  {p.name}")
print()
saved_tables = sorted(p for p in TABLE_DIR.iterdir() if p.suffix == ".csv")
print(f"{len(saved_tables)} tables under {TABLE_DIR}:")
for p in saved_tables:
    print(f"  {p.name}")
""")


# =============================================================================
# Build the .ipynb
# =============================================================================

def build() -> None:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for kind, src in CELLS:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": src.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT} ({len(CELLS)} cells)")


def execute() -> None:
    """Run the notebook in-place so cells acquire their outputs."""
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
        "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=900",
        str(OUT),
    ]
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--execute", action="store_true",
                    help="Run the notebook end-to-end after building.")
    args = ap.parse_args()
    build()
    if args.execute:
        execute()
