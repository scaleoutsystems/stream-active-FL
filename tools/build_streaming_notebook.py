"""Generate notebooks/01_streaming_analysis.ipynb from a structured cell list.

Run as ``python tools/build_streaming_notebook.py`` from the project
root.  Add ``--execute`` to also run the resulting notebook end-to-end
so the cells contain their outputs (~3 minutes wall clock with 24
variants across 3 seeds).

Re-runs without ``--execute`` overwrite cell *source* but produce a
notebook with empty outputs.  Use with care: if the notebook has been
executed and you only need to refresh structure, prefer editing the
cell list and re-executing afterwards.
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


# Each entry is (cell_type, source).  Markdown cells use single-string
# bodies so the resulting ipynb stays readable in diff tools.
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

This notebook produces every table and figure that backs the streaming
chapter of the write-up.  Heavy lifting lives in two reusable modules:

- `streaming_analysis.py` — variant registry, headline tables, per-domain
  grids, iso-accept pairings, ablation comparisons.
- `streaming_figures.py` — accept-rate dynamics, per-domain heatmaps,
  trajectories, ablation bars.

Outputs:

- Tables  -> `reports/streaming/tables/*.csv`
- Figures -> `reports/streaming/figures/*.{pdf,png}`

The headline narrative leads with **accept dynamics** (does the filter
actually route compute toward novel domains?) and **per-domain
performance** (does it improve mAP where it matters?).  Total-val mAP
is included as supporting evidence rather than the headline because
it can mask domain-level wins.

Re-run end to end after a new batch of experiments; everything is keyed
off `outputs/streaming/<variant>/seed_<N>/<timestamp>/`.
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

#: Tail-k for smoothed mAP.  5 matches the round-3/4 reports.
TAIL_K = 5

sa.prime_registry(project_root=PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")
print(f"Figures -> {FIG_DIR}")
print(f"Tables  -> {TABLE_DIR}")
print(f"Variants registered: {len(sa.FEATURED_VARIANTS)}")
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

# Resolve manifest + bootstrap once so accept-dynamics and trajectory
# panels share the same boundaries.
code("""\
sample_dir = sa.latest_seed_dir("no_filter_cityday_curated", 42, project_root=PROJECT_ROOT)
sample_cfg = ah.load_run_config(sample_dir) if sample_dir else {}
MAN_CUR = ah.load_manifest(PROJECT_ROOT, sample_cfg.get("manifest_path") if sample_cfg else None)
BOOT_CUR = ah.get_bootstrap_size(MAN_CUR, sample_cfg)
BOUNDS_CUR, MIDPOINTS_CUR = sa.block_boundaries_and_midpoints(MAN_CUR, bootstrap_frames=BOOT_CUR)

# Composition panels: time-of-day + road-condition stacked area.
ACCEPT_WINDOW = 500  # frames per bucket; defined here so composition matches accept-rate granularity
COMPOSITION_CUR = sa.stream_composition(
    MAN_CUR, bootstrap_frames=BOOT_CUR,
    fields=("time_of_day", "road_condition"),
    window=ACCEPT_WINDOW,
    field_orders={
        "time_of_day": ("day", "twilight", "night"),
        "road_condition": ("normal", "wet", "snow"),
    },
)
TOD_PALETTE = {"day": ah.TOD_COLORS.get("day", "#FFD700"),
               "twilight": ah.TOD_COLORS.get("twilight", "#FF8C00"),
               "night": ah.TOD_COLORS.get("night", "#191970")}
RC_PALETTE = {"normal": "#cccccc",
              "wet": ah.WEATHER_COLORS.get("rain_wet", "#4682B4"),
              "snow": "#88c0d0"}
COMP_PALETTES = {"time_of_day": TOD_PALETTE, "road_condition": RC_PALETTE}
COMP_TITLES = {"time_of_day": "Time-of-day composition of stream window",
               "road_condition": "Road-condition composition of stream window"}
print("Curated stream:", len(BOUNDS_CUR) - 1, "blocks; boundaries =", BOUNDS_CUR)
""")

# =============================================================================
# 3. Accept dynamics --- the headline story
# =============================================================================

md("""\
## 3 Accept dynamics

This section answers *does the filter route compute toward novel
domains?*.  Two complementary views:

- **Per-block accept rate** — how does each variant divide its label
  budget across the 12 manifest blocks?  Random should be flat at its
  global rate; an "intelligent" filter should over-accept on novel
  blocks and under-accept on familiar ones.
- **Accept rate per stream window** — same data along the time axis,
  with stacked-area composition panels showing what *kind* of frames
  the stream is presenting in each window (time-of-day +
  road-condition).""")

code("""\
ROUTING_VARIANTS = [
    "random_p21_cityday_curated",
    "random_p33_cityday_curated",
    "static_p20_cityday_curated",
    "adaptive_window_p20_cityday_curated",
    "adaptive_window_p20_twoRef_cityday_curated",
    "adaptive_window_p20_noBoot_cityday_curated",
    "adaptive_reservoir_p20_cityday_curated",
    "adaptive_reservoir_p20_twoRef_cityday_curated",
    "adaptive_reservoir_p20_noBoot_cityday_curated",
]
routing = sa.per_block_routing(ROUTING_VARIANTS, project_root=PROJECT_ROOT)
routing.to_csv(TABLE_DIR / "per_block_routing_curated.csv")
display(routing.round(3))
fig, _ = sf.plot_per_block_routing(
    routing, baseline_label="rand_p33",
    title="Per-block accept rate — curated stream (mean over seeds)")
sf.save_figure(fig, "01_per_block_routing_curated", out_dir=FIG_DIR)
plt.show()
""")

md("""\
We drop random from the line plot here on purpose: the four adaptive
variants we compare against operate at empirical rates 0.20–0.34, and
pinning a single random line to all of them would be misleading.  The
random row in the per-block routing heatmap above already carries the
"would have been roughly flat" reference.""")

code("""\
ADAPTIVE_DYN_VARIANTS = [
    "adaptive_window_p20_cityday_curated",
    "adaptive_window_p20_twoRef_cityday_curated",
    "adaptive_reservoir_p20_cityday_curated",
    "adaptive_reservoir_p20_twoRef_cityday_curated",
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
    window=ACCEPT_WINDOW,
    title="Accept rate per stream window — adaptive filters (curated)")
sf.save_figure(fig, "02_accept_rate_with_composition_curated", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 3c Accept-rate dynamics — static filters (curated)

For comparison, the static distribution filter (reference fixed at
bootstrap, no refresh) accepts close to its threshold-implied rate
across the entire stream.  Both static variants saturate near 1.0 on
the novel night/twilight blocks: their fixed reference cannot
distinguish "novel" from "everything", so they let nearly all such
frames through and stop being budget-controlled.""")

code("""\
STATIC_DYN_VARIANTS = [
    "static_p15_cityday_curated",
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
    window=ACCEPT_WINDOW,
    title="Accept rate per stream window — static filters (curated)")
sf.save_figure(fig, "02b_accept_rate_with_composition_static_curated", out_dir=FIG_DIR)
plt.show()
""")

# =============================================================================
# 4. Per-domain performance --- the headline story
# =============================================================================

md("""\
## 4 Per-domain performance

This section answers *does the filter improve mAP where it matters?*.
We **prefer per-domain mAP** over total-val mAP for the headline
because total-val mAP can mask both wins on novel domains and losses on
saturated ones; per-domain numbers are more interpretable and harder to
fool with budget effects.""")

md("### 4a Per-domain end-of-stream mAP (curated)")

code("""\
cur_variants = [v for v in sa.FEATURED_VARIANTS if sa.manifest_for_variant(v) == "curated"]
tmp_variants = [v for v in sa.FEATURED_VARIANTS if sa.manifest_for_variant(v) == "temporal"]

grid_cur = sa.per_domain_grid(cur_variants, project_root=PROJECT_ROOT, tail_k=TAIL_K)
grid_cur.to_csv(TABLE_DIR / "per_domain_curated.csv")
fig, _ = sf.plot_per_domain_heatmap(
    grid_cur, title="Per-domain mAP (smoothed tail-5) — curated stream")
sf.save_figure(fig, "03_per_domain_heatmap_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 4b Per-block delta vs iso-accept random (curated)")

code("""\
delta_grid = sa.per_domain_delta_grid(project_root=PROJECT_ROOT, tail_k=TAIL_K)
delta_summary = sa.per_domain_delta_summary(project_root=PROJECT_ROOT, tail_k=TAIL_K)
delta_grid.to_csv(TABLE_DIR / "per_domain_delta.csv")
delta_summary.to_csv(TABLE_DIR / "per_domain_delta_summary.csv", index=False)
display(delta_summary.round(4))
fig, _ = sf.plot_per_domain_delta_heatmap(
    delta_grid,
    title=r"Per-domain $\\Delta$mAP (filter $-$ iso-accept random)")
sf.save_figure(fig, "04_per_domain_delta_heatmap", out_dir=FIG_DIR)
plt.show()
""")

md("### 4c Balanced vs worst-block mAP (curated)")

code("""\
summary_cur = sa.per_domain_summary(cur_variants, project_root=PROJECT_ROOT, tail_k=TAIL_K)
summary_tmp = sa.per_domain_summary(tmp_variants, project_root=PROJECT_ROOT, tail_k=TAIL_K)
summary_cur.to_csv(TABLE_DIR / "per_domain_summary_curated.csv", index=False)
summary_tmp.to_csv(TABLE_DIR / "per_domain_summary_temporal.csv", index=False)
display(summary_cur.round(4))
fig, _ = sf.plot_balanced_vs_worst(
    summary_cur,
    title="Balanced vs worst-block mAP — curated stream")
sf.save_figure(fig, "05_balanced_vs_worst_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 4d Per-block mAP trajectory through the stream (curated)")

code("""\
TRAJ_VARIANTS_CUR = [
    "no_filter_cityday_curated",
    "random_p33_cityday_curated",
    "adaptive_window_p20_cityday_curated",
    "adaptive_window_p20_twoRef_cityday_curated",
    "adaptive_reservoir_p20_cityday_curated",
]
NOVEL_BLOCKS = [
    "city_night",
    "city_twilight",
    "highway_twi-night",
    "arterial-rural_twi-night",
    "arterial-urban_twi-night",
    "smaller-rural_all",
]
trajectories_cur = {
    v: sa.per_domain_trajectory(v, NOVEL_BLOCKS, project_root=PROJECT_ROOT)
    for v in TRAJ_VARIANTS_CUR
}
trajectories_cur = {v: df for v, df in trajectories_cur.items() if not df.empty}
fig, _ = sf.plot_per_block_trajectory(
    trajectories_cur, NOVEL_BLOCKS,
    x_col="items_processed", n_cols=3,
    title="Per-block mAP trajectory — curated (novel/late blocks)")
sf.save_figure(fig, "06_per_block_trajectory_curated", out_dir=FIG_DIR)
plt.show()
""")

# =============================================================================
# 5. Cross-stream-order replication (temporal manifest)
# =============================================================================

md("""\
## 5 Cross-stream-order replication (temporal manifest)

The temporal manifest streams chronologically rather than in curated
domain blocks.  If the filter genuinely tracks novelty, the same kind
of accept-rate dynamics and per-domain wins should reproduce on the
temporal stream — without us hand-curating the block order.""")

code("""\
sample_dir_tmp = sa.latest_seed_dir("no_filter_cityday_temporal", 42, project_root=PROJECT_ROOT)
sample_cfg_tmp = ah.load_run_config(sample_dir_tmp) if sample_dir_tmp else {}
MAN_TMP = ah.load_manifest(PROJECT_ROOT, sample_cfg_tmp.get("manifest_path") if sample_cfg_tmp else None)
BOOT_TMP = ah.get_bootstrap_size(MAN_TMP, sample_cfg_tmp)
BOUNDS_TMP, MIDPOINTS_TMP = sa.block_boundaries_and_midpoints(MAN_TMP, bootstrap_frames=BOOT_TMP)
COMPOSITION_TMP = sa.stream_composition(
    MAN_TMP, bootstrap_frames=BOOT_TMP,
    fields=("time_of_day", "road_condition"),
    window=ACCEPT_WINDOW,
    field_orders={
        "time_of_day": ("day", "twilight", "night"),
        "road_condition": ("normal", "wet", "snow"),
    },
)
print(f"Temporal stream: {len(BOUNDS_TMP) - 1} blocks; bootstrap={BOOT_TMP}")
""")

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
    window=ACCEPT_WINDOW,
    title="Accept rate per stream window — adaptive filters (temporal)")
sf.save_figure(fig, "07_accept_rate_with_composition_temporal", out_dir=FIG_DIR)
plt.show()
""")

code("""\
grid_tmp = sa.per_domain_grid(tmp_variants, project_root=PROJECT_ROOT, tail_k=TAIL_K)
grid_tmp.to_csv(TABLE_DIR / "per_domain_temporal.csv")
fig, _ = sf.plot_per_domain_heatmap(
    grid_tmp, title="Per-domain mAP (smoothed tail-5) — temporal stream")
sf.save_figure(fig, "08_per_domain_heatmap_temporal", out_dir=FIG_DIR)
plt.show()
""")

code("""\
TRAJ_VARIANTS_TMP = [
    "no_filter_cityday_temporal",
    "random_p28_cityday_temporal",
    "adaptive_window_p20_cityday_temporal",
    "adaptive_window_p20_twoRef_cityday_temporal",
    "adaptive_reservoir_p20_cityday_temporal",
]
trajectories_tmp = {
    v: sa.per_domain_trajectory(v, NOVEL_BLOCKS, project_root=PROJECT_ROOT)
    for v in TRAJ_VARIANTS_TMP
}
trajectories_tmp = {v: df for v, df in trajectories_tmp.items() if not df.empty}
fig, _ = sf.plot_per_block_trajectory(
    trajectories_tmp, NOVEL_BLOCKS,
    x_col="items_processed", n_cols=3,
    title="Per-block mAP trajectory — temporal (novel/late blocks)")
sf.save_figure(fig, "09_per_block_trajectory_temporal", out_dir=FIG_DIR)
plt.show()
""")

# =============================================================================
# 6. Methodology ablations
# =============================================================================

md("""\
## 6 Methodology ablations

Three controlled comparisons that justify the design choices:

- **Static vs adaptive**: bootstrap-only Mahalanobis vs periodic refresh.
- **Two-reference vs single-reference**: keeping the bootstrap Gaussian
  *alongside* the adaptive Gaussian instead of replacing it.
- **Bootstrap-anchor (noBoot)**: zeroing out the bootstrap anchor; tests
  whether the anchor is doing real work.

The within-refresh accept-rate trace at the end is the diagnostic that
originally motivated the two-reference design.""")

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
    two_ref,
    title=r"Two-reference vs single-reference (Δ = twoRef $-$ single)")
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
    no_boot,
    title=r"Bootstrap anchor (Δ = noBoot $-$ anchored)")
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
    seg, title="Per-refresh accept rate — curated (seed 42)")
sf.save_figure(fig, "13_refresh_segment_decay_curated", out_dir=FIG_DIR)
plt.show()
""")

# =============================================================================
# 7. Iso-accept fairness (supporting evidence)
# =============================================================================

md("""\
## 7 Iso-accept fairness (supporting)

Headline-style scatter that reads "filter beats iso-accept random by
ΔmAP at this accept rate".  Used as supporting evidence rather than the
main story because the per-domain heatmaps in §4 carry strictly more
information.""")

code("""\
iso = sa.iso_accept_table(project_root=PROJECT_ROOT, tail_k=TAIL_K)
iso.to_csv(TABLE_DIR / "iso_accept.csv", index=False)
display_cols = ["filter_label", "manifest", "filter_accept", "filter_smoothed",
                "random_label", "random_accept", "random_smoothed",
                "accept_gap", "delta_smoothed"]
iso[display_cols].round(4)
""")

code("""\
fig, _ = sf.plot_inventory_scatter(inv, manifest="curated")
sf.save_figure(fig, "14_iso_accept_curated", out_dir=FIG_DIR)
plt.show()
fig, _ = sf.plot_iso_accept_scatter(iso, manifest="curated")
sf.save_figure(fig, "15_iso_accept_delta_curated", out_dir=FIG_DIR)
plt.show()
""")

# =============================================================================
# 8. Supporting analyses (per-class, forgetting, compute, total mAP)
# =============================================================================

md("""\
## 8 Supporting analyses

These figures sit in the chapter appendix or get cited inline as
diagnostics.  Total-val mAP trajectories are placed here, behind the
per-domain story, on purpose.""")

md("### 8a Per-class breakdown")

code("""\
PER_CLASS_VARIANTS = [
    "no_filter_cityday_curated",
    "random_p33_cityday_curated",
    "static_p20_cityday_curated",
    "adaptive_window_p20_cityday_curated",
    "adaptive_window_p20_twoRef_cityday_curated",
    "adaptive_reservoir_p20_cityday_curated",
    "adaptive_reservoir_p20_twoRef_cityday_curated",
]
per_class = sa.per_class_grid(PER_CLASS_VARIANTS, project_root=PROJECT_ROOT, tail_k=TAIL_K)
per_class.to_csv(TABLE_DIR / "per_class_curated.csv")
display(per_class.round(4))
fig, _ = sf.plot_per_class_heatmap(
    per_class,
    title="Per-class end-of-stream AP (smoothed tail-5) — curated")
sf.save_figure(fig, "A1_per_class_heatmap_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 8b Forgetting analysis (early vs late stream)")

code("""\
forget = sa.forgetting_table(
    PER_CLASS_VARIANTS, project_root=PROJECT_ROOT, n_bins=4)
forget.to_csv(TABLE_DIR / "forgetting_curated.csv", index=False)
fig, _ = sf.plot_forgetting_heatmap(
    forget, metric="delta",
    title=r"Forgetting: $\\Delta$ AP (last quartile $-$ first quartile)")
sf.save_figure(fig, "A2_forgetting_delta_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 8c Compute efficiency (steps to reach a target mAP)")

code("""\
TARGETS = [0.20, 0.22, 0.24]
EFFICIENCY_VARIANTS = [
    "no_filter_cityday_curated",
    "random_p33_cityday_curated",
    "random_p77_cityday_curated",
    "static_p20_cityday_curated",
    "adaptive_window_p20_cityday_curated",
    "adaptive_window_p20_twoRef_cityday_curated",
    "adaptive_reservoir_p20_cityday_curated",
]
steps_table = sa.steps_to_reach_table(
    EFFICIENCY_VARIANTS, TARGETS,
    project_root=PROJECT_ROOT, x_col="optimizer_steps")
steps_table.to_csv(TABLE_DIR / "steps_to_reach.csv", index=False)
display(steps_table.pivot(index="target_mAP", columns="label",
                          values="optimizer_steps").round(0))
fig, _ = sf.plot_steps_to_target(
    steps_table, x_col="optimizer_steps",
    title="Optimizer steps to reach a target mAP — curated")
sf.save_figure(fig, "A3_steps_to_target_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 8d Overall validation mAP through the stream")

code("""\
TRAJ_HEADLINE = [
    "no_filter_cityday_curated",
    "random_p77_cityday_curated",
    "static_p20_cityday_curated",
    "adaptive_window_p20_cityday_curated",
    "adaptive_window_p20_twoRef_cityday_curated",
    "adaptive_reservoir_p20_cityday_curated",
]
TRAJ_HEADLINE = [v for v in TRAJ_HEADLINE if v in sa.FEATURED_VARIANTS]
traj_cur = sa.mAP_trajectory(TRAJ_HEADLINE, project_root=PROJECT_ROOT)
block_trans = list(zip(BOUNDS_CUR, [m[1] for m in MIDPOINTS_CUR] + [""]))
fig, _ = sf.plot_overall_mAP_trajectory(
    traj_cur, x_col="items_processed", block_transitions=block_trans,
    title="Overall val mAP through the curated stream")
sf.save_figure(fig, "A4_overall_mAP_trajectory_curated", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 16. Index of saved figures
# =============================================================================

md("## 16 Saved figures and tables index")

code("""\
import os
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
