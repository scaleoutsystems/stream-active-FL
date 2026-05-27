"""Generate notebooks/02_federated_analysis.ipynb from a structured cell list.

Run as ``python tools/build_federated_notebook.py`` from the project
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
       / "notebooks" / "02_federated_analysis.ipynb")


CELLS: List[Tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.lstrip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.lstrip("\n").rstrip() + "\n"))


# =============================================================================
# 1. Title and setup
# =============================================================================

md("""\
# Federated detection — analysis

Tables and figures for federated detection.

Outputs:

- Tables  -> `reports/federated/tables/*.csv`
- Figures -> `reports/federated/figures/*.{pdf,png}`
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

# Make the package importable when stream-active-fl has not been
# installed (e.g. when running the notebook from a project clone).
_proj = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
if str(_proj / "src") not in sys.path:
    sys.path.insert(0, str(_proj / "src"))

from stream_active_fl.analysis import runs as ah
from stream_active_fl.analysis import federated as fa
from stream_active_fl.analysis import streaming as sa
from stream_active_fl.analysis.figures import federated as ff

ah.setup_notebook_style(dpi=120)

PROJECT_ROOT = ah.find_project_root()
OUTPUTS = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports" / "federated"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

#: Tail-k for smoothed mAP (matches the streaming chapter).
TAIL_K = 5
#: Smoothing window (in rounds) for trajectory plots.  Picked to be
#: large enough to dampen single-checkpoint spikes (the worst seed
#: dips ~0.02 mAP) but small enough to preserve trends.
TRAJ_SMOOTHING = 5

fa.prime_registry(project_root=PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")
print(f"Variants registered: {len(fa.FEATURED_VARIANTS)}")
""")

md("""\
### 1a Headline variants and pairs
""")

code("""\
HEADLINE_VARIANTS_DEFAULT_CURATED = [
    "fed_no_filter_cityday_curated",
    "fed_static_p20_cityday_curated",
    "fed_adaptive_window_p20_twoRef_cityday_curated",
    "fed_adaptive_reservoir_p20_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    "fed_random_p15_cityday_curated",
    "fed_random_p18_cityday_curated",
    "fed_random_p77_cityday_curated",
]

# Iso-accept pairs used for the per-cell delta heatmap (sec 3a/3b).
HEADLINE_PAIRS_DEFAULT_CURATED = [
    ("fed_static_p20_cityday_curated",                    "fed_random_p77_cityday_curated"),
    ("fed_adaptive_window_p20_twoRef_cityday_curated",    "fed_random_p12_cityday_curated"),
    ("fed_adaptive_reservoir_p20_cityday_curated",        "fed_random_p18_cityday_curated"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_curated", "fed_random_p15_cityday_curated"),
]
HEADLINE_PAIRS_DEFAULT_TEMPORAL = [
    ("fed_adaptive_reservoir_p15_twoRef_cityday_temporal", "fed_random_p15_cityday_temporal"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_temporal", "fed_random_p19_cityday_temporal"),
]
HEADLINE_PAIRS_HL_CURATED = [
    ("fed_adaptive_reservoir_p15_twoRef_cityday_curated_heavyLocal", "fed_random_p21_cityday_curated_heavyLocal"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal", "fed_random_p26_cityday_curated_heavyLocal"),
]
HEADLINE_PAIRS_HL_TEMPORAL = [
    ("fed_adaptive_reservoir_p15_twoRef_cityday_temporal_heavyLocal", "fed_random_p25_cityday_temporal_heavyLocal"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_temporal_heavyLocal", "fed_random_p30_cityday_temporal_heavyLocal"),
]

# One row per cell of the 2 x 2 design, taken at the cell's headline
# (filter, random) pair.  Used by the delta-heatmap grids and the
# schedule x partition mechanism table.
CELL_HEADLINE_PAIRS = [
    ("default + curated",
     "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
     "fed_random_p15_cityday_curated"),
    ("default + temporal",
     "fed_adaptive_reservoir_p20_twoRef_cityday_temporal",
     "fed_random_p19_cityday_temporal"),
    ("heavy-local + curated",
     "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
     "fed_random_p26_cityday_curated_heavyLocal"),
    ("heavy-local + temporal",
     "fed_adaptive_reservoir_p20_twoRef_cityday_temporal_heavyLocal",
     "fed_random_p30_cityday_temporal_heavyLocal"),
]

# Short cell-tag labels for the delta heatmaps.  ``per_domain_delta_grid``
# emits one column per (filter, random) pairing keyed by
# ``"{filter_label} - {random_label}"``; the per-block trajectory delta
# table (`per_block_trajectory_delta`) emits one row per pairing
# keyed only by ``filter_label``.  We provide both rename maps so the
# heatmap headers collapse to ``default+curated`` etc. in print.
DELTA_GRID_COLUMN_RENAME = {
    f"{fa.label_for(f)} - {fa.label_for(r)}": tag
    for (tag, f, r) in CELL_HEADLINE_PAIRS
}
TRAJ_DELTA_COLUMN_RENAME = {
    fa.label_for(f): tag for (tag, f, _) in CELL_HEADLINE_PAIRS
}
DELTA_COLUMN_ORDER = [tag for (tag, _, _) in CELL_HEADLINE_PAIRS]
""")


# =============================================================================
# 2. Inventory
# =============================================================================

md("""\
## 2 Variant inventory

One row per variant with empirical accept rate, smoothed tail-`TAIL_K`
mAP across 3 seeds, and total compute (items processed, optimizer
steps), plus a `schedule` column distinguishing default (30 rounds
x 1 000 items) from `heavyLocal` (10 rounds x 3 000 items).""")

code("""\
inv = fa.inventory_table(project_root=PROJECT_ROOT, tail_k=TAIL_K)
inv.to_csv(TABLE_DIR / "inventory.csv", index=False)
display_cols = [
    "label", "manifest", "schedule", "family", "n_seeds", "n_rounds",
    "accept_rate", "smoothed_mAP", "smoothed_std",
    "items_processed", "optimizer_steps",
]
inv[display_cols].round(4)
""")


# =============================================================================
# 3. Iso-accept leaderboard
# =============================================================================

md("""\
## 3 Iso-accept leaderboard (fig 00)
""")

code("""\
# Restrict each panel to the cell's headline variants so the labels
# do not overlap.  Tighter-accept (tau_10/15) and ablation-only
# variants belong on figs 11-13 and the iso_accept.csv table; here
# we want the four families compared at one (filter, random) pair
# per family, plus the no-filter ceiling and (in default+curated)
# the static dual-axis reference.  All randoms in the cell remain
# visible as the iso-accept envelope.
ISO_HEADLINE_BY_CELL = {
    ("curated", "default"): [
        "fed_no_filter_cityday_curated",
        "fed_static_p20_cityday_curated",
        "fed_adaptive_window_p20_twoRef_cityday_curated",
        "fed_adaptive_reservoir_p20_cityday_curated",
        "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    ],
    ("temporal", "default"): [
        "fed_no_filter_cityday_temporal",
        "fed_adaptive_reservoir_p15_twoRef_cityday_temporal",
        "fed_adaptive_reservoir_p20_twoRef_cityday_temporal",
    ],
    ("curated", "heavyLocal"): [
        "fed_no_filter_cityday_curated_heavyLocal",
        "fed_adaptive_window_p20_twoRef_cityday_curated_heavyLocal",
        "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal",
        "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
    ],
    ("temporal", "heavyLocal"): [
        "fed_no_filter_cityday_temporal_heavyLocal",
        "fed_adaptive_reservoir_p15_twoRef_cityday_temporal_heavyLocal",
        "fed_adaptive_reservoir_p20_twoRef_cityday_temporal_heavyLocal",
    ],
}
ISO_CELLS = [
    ("00a_iso_accept_default_curated",      "curated",  "default"),
    ("00b_iso_accept_default_temporal",     "temporal", "default"),
    ("00c_iso_accept_heavyLocal_curated",   "curated",  "heavyLocal"),
    ("00d_iso_accept_heavyLocal_temporal",  "temporal", "heavyLocal"),
]
for slug, manifest, schedule in ISO_CELLS:
    fig, _ = ff.plot_inventory_scatter(
        inv,
        manifest=manifest,
        schedule=schedule,
        headline_variants=ISO_HEADLINE_BY_CELL[(manifest, schedule)],
        annotate_filters_only=True,
    )
    ff.save_figure(fig, slug, out_dir=FIG_DIR)
    plt.show()
""")

md("### 3a Iso-accept pairings table")

code("""\
iso = fa.iso_accept_table(project_root=PROJECT_ROOT, tail_k=TAIL_K)
iso.to_csv(TABLE_DIR / "iso_accept.csv", index=False)
iso_cols = [
    "filter_label", "random_label", "schedule",
    "filter_accept", "random_accept", "accept_gap",
    "filter_smoothed", "random_smoothed", "delta_smoothed",
]
iso[iso_cols].round(4)
""")


# =============================================================================
# 4. Per-client compute routing (figs 01-07)
# =============================================================================

md("""\
## 4 Per-client compute routing (figs 01-07)
""")

md("""\
### 4.1 Per-client stream composition (figs 01a-01b)
""")

code("""\
# Per-client aggregate fractions -> CSV (used for chapter text numbers).
COMP_CURATED = fa.per_client_dimension_composition(
    "fed_no_filter_cityday_curated", project_root=PROJECT_ROOT)
COMP_TMP = fa.per_client_dimension_composition(
    "fed_no_filter_cityday_temporal", project_root=PROJECT_ROOT)
for tag, comp in (("curated", COMP_CURATED), ("temporal", COMP_TMP)):
    rows = []
    for field, df in comp.items():
        if df is None or df.empty:
            continue
        long = df.reset_index().melt(id_vars="client", var_name="bucket",
                                     value_name="fraction")
        long["field"] = field
        rows.append(long)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            TABLE_DIR / f"per_client_composition_{tag}.csv", index=False)

# Load manifests and derive stream-composition data for both partitions.
_sample_dir = fa.latest_seed_dir("fed_no_filter_cityday_curated", project_root=PROJECT_ROOT)
_sample_cfg = ah.load_run_config(_sample_dir) if _sample_dir else {}
MAN_FED_CUR = ah.load_manifest(PROJECT_ROOT, _sample_cfg.get("manifest_path") if _sample_cfg else None)
assert MAN_FED_CUR is not None, "Curated federated manifest not found"
BOOT_FED_CUR = ah.get_bootstrap_size(MAN_FED_CUR, _sample_cfg)
BOUNDS_FED_CUR, _ = sa.block_boundaries_and_midpoints(MAN_FED_CUR, bootstrap_frames=BOOT_FED_CUR)

_sample_dir = fa.latest_seed_dir("fed_no_filter_cityday_temporal", project_root=PROJECT_ROOT)
_sample_cfg = ah.load_run_config(_sample_dir) if _sample_dir else {}
MAN_FED_TMP = ah.load_manifest(PROJECT_ROOT, _sample_cfg.get("manifest_path") if _sample_cfg else None)
assert MAN_FED_TMP is not None, "Temporal federated manifest not found"
BOOT_FED_TMP = ah.get_bootstrap_size(MAN_FED_TMP, _sample_cfg)
BOUNDS_FED_TMP, _ = sa.block_boundaries_and_midpoints(MAN_FED_TMP, bootstrap_frames=BOOT_FED_TMP)

FED_ACCEPT_WINDOW = 500

def _fed_weather_bucket(frame):
    w = (frame.get("scraped_weather") or "").lower()
    rc = (frame.get("road_condition") or "").lower()
    if "snow" in w or "snow" in rc:
        return "snow"
    if "rain" in w or "wet" in rc:
        return "rain_wet"
    if "cloud" in w or "fog" in w or "overcast" in w:
        return "cloudy"
    return "clear"

FED_COMP_ORDERS = {
    "time_of_day": ("day", "twilight", "night"),
    "road_type":   ("city", "arterial-urban", "highway", "arterial-rural", "smaller-rural"),
    "weather":     ("clear", "cloudy", "rain_wet", "snow"),
}
FED_COMP_PALETTES = {
    "time_of_day": {"day":      ah.TOD_COLORS["day"],
                    "twilight": ah.TOD_COLORS["twilight"],
                    "night":    ah.TOD_COLORS["night"]},
    "road_type":   {"city":           "#1f78b4",
                    "arterial-urban": "#a6cee3",
                    "highway":        "#fdbf6f",
                    "arterial-rural": "#b2df8a",
                    "smaller-rural":  "#33a02c"},
    "weather":     {"clear":    ah.WEATHER_COLORS["clear"],
                    "cloudy":   ah.WEATHER_COLORS["cloudy"],
                    "rain_wet": ah.WEATHER_COLORS["rain_wet"],
                    "snow":     ah.WEATHER_COLORS["snow"]},
}
FED_COMP_TITLES = {
    "time_of_day": "Time-of-day composition of stream window",
    "road_type":   "Road-type composition of stream window",
    "weather":     "Weather composition of stream window",
}

COMP_STREAM_CUR = sa.stream_composition(
    MAN_FED_CUR, bootstrap_frames=BOOT_FED_CUR,
    fields=("time_of_day", "road_type", "weather"),
    window=FED_ACCEPT_WINDOW,
    field_orders=FED_COMP_ORDERS,
    field_derivers={"weather": _fed_weather_bucket},
)
CLIENT_PART_CUR = fa.curated_client_boundaries(MAN_FED_CUR)
fig, _ = ff.plot_stream_composition_with_partitions(
    COMP_STREAM_CUR,
    block_boundaries=BOUNDS_FED_CUR,
    client_boundaries=CLIENT_PART_CUR,
    composition_palettes=FED_COMP_PALETTES,
    composition_titles=FED_COMP_TITLES,
)
ff.save_figure(fig, "01a_per_client_composition_curated", out_dir=FIG_DIR)
plt.show()

COMP_STREAM_TMP = sa.stream_composition(
    MAN_FED_TMP, bootstrap_frames=BOOT_FED_TMP,
    fields=("time_of_day", "road_type", "weather"),
    window=FED_ACCEPT_WINDOW,
    field_orders=FED_COMP_ORDERS,
    field_derivers={"weather": _fed_weather_bucket},
)
CLIENT_PART_TMP = fa.temporal_client_boundaries(MAN_FED_TMP, bootstrap_frames=BOOT_FED_TMP)
fig, _ = ff.plot_stream_composition_with_partitions(
    COMP_STREAM_TMP,
    block_boundaries=BOUNDS_FED_TMP,
    client_boundaries=CLIENT_PART_TMP,
    composition_palettes=FED_COMP_PALETTES,
    composition_titles=FED_COMP_TITLES,
)
ff.save_figure(fig, "01b_per_client_composition_temporal", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 4.2 Per-block accept rate (fig 02)
""")

code("""\
PB_FILTER_VARIANTS = [
    "fed_adaptive_window_p20_twoRef_cityday_curated",
    "fed_adaptive_reservoir_p20_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
]
PB_STATIC = "fed_static_p20_cityday_curated"
pb = fa.per_block_accept_rate_table(
    PB_FILTER_VARIANTS + [PB_STATIC], project_root=PROJECT_ROOT)
pb.to_csv(TABLE_DIR / "per_block_accept_rate.csv", index=False)
# Order matches the streaming manifest's CITYDAY_CURATED_ORDER (the
# truth source for the curated stream sequence) so this figure aligns
# left-to-right with the streaming per-block routing figure.  Within
# city_day, the order is largest-first (cloudy ~22.9k -> clear ~7.6k
# -> rain_wet ~7.1k -> snow ~1.0k frames).
PB_BLOCK_ORDER = [
    "city_day_cloudy", "city_day_clear",
    "city_day_rain_wet", "city_day_snow", "city_twilight", "city_night",
    "arterial-urban_day", "arterial-urban_twi-night",
    "highway_day", "highway_twi-night",
    "arterial-rural_day", "arterial-rural_twi-night", "smaller-rural_all",
]
PB_BLOCK_ORDER = [b for b in PB_BLOCK_ORDER if b in pb["block"].unique()]
fig, _ = ff.plot_per_block_accept_rate(
    pb,
    filter_variants=PB_FILTER_VARIANTS,
    static_variant=PB_STATIC,
    block_order=PB_BLOCK_ORDER,
    show_std=True,
    ymax=0.45,
)
ff.save_figure(fig, "02_per_block_accept_rate_default_curated", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 4.3 Per-round per-client accept-rate dynamics (figs 03a-03b)
""")

code("""\
PR_CURATED_VARIANTS = [
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    "fed_adaptive_reservoir_p20_cityday_curated",
    "fed_adaptive_window_p20_twoRef_cityday_curated",
    "fed_static_p20_cityday_curated",
]
PR_TMP_VARIANTS = [
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal",
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal",
]
pr = fa.per_round_per_client_accept(
    PR_CURATED_VARIANTS + PR_TMP_VARIANTS,
    project_root=PROJECT_ROOT,
)
pr.to_csv(TABLE_DIR / "per_round_per_client_accept.csv", index=False)

PR_CURATED_PANELS = [
    ("Reservoir two-ref",
     pr[pr["variant"] == "fed_adaptive_reservoir_p20_twoRef_cityday_curated"]),
    ("Reservoir single-ref",
     pr[pr["variant"] == "fed_adaptive_reservoir_p20_cityday_curated"]),
    ("Window two-ref",
     pr[pr["variant"] == "fed_adaptive_window_p20_twoRef_cityday_curated"]),
    ("Static",
     pr[pr["variant"] == "fed_static_p20_cityday_curated"]),
]
fig, _ = ff.plot_per_round_per_client_accept(
    PR_CURATED_PANELS, n_cols=2, smoothing_window=3, show_std=True)
ff.save_figure(fig, "03a_per_round_per_client_accept_default_curated",
               out_dir=FIG_DIR)
plt.show()

PR_TMP_PANELS = [
    ("Reservoir two-ref",
     pr[pr["variant"] == "fed_adaptive_reservoir_p20_twoRef_cityday_temporal"]),
]
fig, _ = ff.plot_per_round_per_client_accept(
    PR_TMP_PANELS, n_cols=1, smoothing_window=3, show_std=True,
    figsize=(7.0, 4.2))
ff.save_figure(fig, "03b_per_round_per_client_accept_default_temporal",
               out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 4.4 Per-cell per-client accept-rate bars (figs 04-06)

Cell-by-cell summary of the per-client routing.  Each panel collapses
all 30 rounds into a single mean accept rate per client; this is the
quick reference to compare filters within a cell.""")

code("""\
per_client = fa.per_client_accept_table(project_root=PROJECT_ROOT)
per_client.to_csv(TABLE_DIR / "per_client_accept.csv", index=False)
display(per_client.head(20).round(4))
""")

md("### 4.4a Per-client accept rate -- default schedule, curated partition (fig 04)")

code("""\
# Static p20 accepts ~0.77 of all frames so its per-client bars
# dwarf the adaptive bars on a shared y-axis (fig 00 already
# captures static's iso-accept position; fig 02 shows its per-block
# routing).  Random is omitted from these per-client bars because
# its bars are flat by construction; the iso-accept comparison is
# better made through fig 00 and the iso_accept.csv table.
ROUTING_VARIANTS_DEFAULT = [
    "fed_adaptive_window_p20_twoRef_cityday_curated",
    "fed_adaptive_reservoir_p20_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
]
fig, _ = ff.plot_per_client_accept_rates(
    per_client, variants=ROUTING_VARIANTS_DEFAULT, ymax=0.30)
ff.save_figure(fig, "04_per_client_accept_default_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 4.4b Per-client accept rate -- heavy-local schedule, curated partition (fig 05)")

code("""\
ROUTING_VARIANTS_HL = [
    "fed_adaptive_window_p20_twoRef_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p15_twoRef_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
]
HL_LABELS = {
    "fed_adaptive_window_p20_twoRef_cityday_curated_heavyLocal":
        "Window two-ref",
    "fed_adaptive_reservoir_p15_twoRef_cityday_curated_heavyLocal":
        r"Reservoir two-ref ($\\tau_{15}$)",
    "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal":
        "Reservoir single-ref",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal":
        "Reservoir two-ref",
}
fig, _ = ff.plot_per_client_accept_rates(
    per_client, variants=ROUTING_VARIANTS_HL,
    ymax=0.40, label_map=HL_LABELS)
ff.save_figure(fig, "05_per_client_accept_heavyLocal_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 4.4c Per-client accept rate -- temporal partition (fig 06)")

code("""\
ROUTING_VARIANTS_TMP = [
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal",
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal_heavyLocal",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal_heavyLocal",
]
# Strip the redundant "(temporal, ...)" qualifier since this whole
# panel is dedicated to the temporal manifest.  Keep "(HL)" so the
# heavy-local pairs remain distinguishable.
TMP_LABELS = {
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal":
        r"Reservoir two-ref ($\\tau_{15}$)",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal":
        "Reservoir two-ref",
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal_heavyLocal":
        r"Reservoir two-ref ($\\tau_{15}$, HL)",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal_heavyLocal":
        "Reservoir two-ref (HL)",
}
fig, _ = ff.plot_per_client_accept_rates(
    per_client, variants=ROUTING_VARIANTS_TMP,
    ymax=0.45, label_map=TMP_LABELS)
ff.save_figure(fig, "06_per_client_accept_temporal", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 4.5 Novelty-routing summary table

`novelty_ratio = mean(C1, C2, C3 accept rate) / C0 accept rate`.
Values > 1 mean the variant routes more compute to novel-domain
clients than to the familiar one; values ~ 1 mean flat routing
(random's behavior by construction).""")

code("""\
novelty = fa.novelty_routing_summary(project_root=PROJECT_ROOT)
novelty.to_csv(TABLE_DIR / "novelty_routing.csv", index=False)
display(novelty.round(4))
""")


# =============================================================================
# 5. Per-block deltas (figs 07-08)
# =============================================================================

md("""\
## 5 Per-block deltas
""")

md("### 5a Per-block tail-`TAIL_K` Δ-mAP heatmap (fig 07)")

code("""\
CELL_PAIRS = [(f, r) for (_, f, r) in CELL_HEADLINE_PAIRS]
delta_grid = fa.per_domain_delta_grid(CELL_PAIRS,
                                      project_root=PROJECT_ROOT, tail_k=TAIL_K)
delta_grid.to_csv(TABLE_DIR / "per_domain_delta_curated.csv")
display(delta_grid.round(4))

delta_summary = fa.per_domain_delta_summary(CELL_PAIRS,
                                            project_root=PROJECT_ROOT, tail_k=TAIL_K)
delta_summary.to_csv(TABLE_DIR / "per_domain_delta_summary.csv", index=False)
display(delta_summary.round(4))

fig, _ = ff.plot_per_domain_delta_heatmap(
    delta_grid,
    column_rename=DELTA_GRID_COLUMN_RENAME,
)
ff.save_figure(fig, "07_per_domain_delta_headline", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 5b Per-block trajectory Δ heatmap (fig 08)
""")

code("""\
traj_delta = fa.per_block_trajectory_delta(CELL_PAIRS,
                                           project_root=PROJECT_ROOT)
traj_delta.to_csv(TABLE_DIR / "per_block_trajectory_delta.csv", index=False)
fig, _ = ff.plot_per_block_trajectory_delta(
    traj_delta,
    metric="cum_avg_delta",
    column_rename=TRAJ_DELTA_COLUMN_RENAME,
    column_order=DELTA_COLUMN_ORDER,
)
ff.save_figure(fig, "08_per_block_trajectory_delta_headline", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 6. Trajectory views (figs 09-10)
# =============================================================================

md("""\
## 6 Trajectory views
""")

md("""\
### 6a Per-category mAP trajectories (figs 09a-09e)
""")

code("""\
CATEGORY_BUCKETS = ["night", "twilight", "wet", "snow"]
def _category_trajs(variants):
    parts = {}
    for v in variants:
        tod = fa.per_block_trajectory(
            v, ["night", "twilight"], dim="time_of_day",
            project_root=PROJECT_ROOT)
        rc = fa.per_block_trajectory(
            v, ["wet", "snow"], dim="road_condition",
            project_root=PROJECT_ROOT)
        pieces = [df for df in (tod, rc) if df is not None and not df.empty]
        if pieces:
            parts[v] = pd.concat(pieces, ignore_index=True)
    return parts

# Iso-accept (filter, random) pairs to render, with the no-filter
# ceiling pulled from the same cell.  Each row becomes one figure.
PER_CATEGORY_FIGURES = [
    ("09a_per_category_trajectory_default_curated_reservoirTwoRef",
     "fed_no_filter_cityday_curated",
     "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
     "fed_random_p15_cityday_curated"),
    ("09b_per_category_trajectory_default_curated_reservoirSingleRef",
     "fed_no_filter_cityday_curated",
     "fed_adaptive_reservoir_p20_cityday_curated",
     "fed_random_p18_cityday_curated"),
    ("09c_per_category_trajectory_default_curated_windowTwoRef",
     "fed_no_filter_cityday_curated",
     "fed_adaptive_window_p20_twoRef_cityday_curated",
     "fed_random_p12_cityday_curated"),
    ("09d_per_category_trajectory_default_curated_static",
     "fed_no_filter_cityday_curated",
     "fed_static_p20_cityday_curated",
     "fed_random_p77_cityday_curated"),
    ("09e_per_category_trajectory_default_temporal_reservoirTwoRef",
     "fed_no_filter_cityday_temporal",
     "fed_adaptive_reservoir_p20_twoRef_cityday_temporal",
     "fed_random_p19_cityday_temporal"),
]
for slug, ceiling, fltr, rnd in PER_CATEGORY_FIGURES:
    cat_trajs = _category_trajs([ceiling, fltr, rnd])
    fig, _ = ff.plot_per_block_trajectory(
        cat_trajs, CATEGORY_BUCKETS,
        n_cols=2,
        smoothing_window=TRAJ_SMOOTHING,
        show_std=True,
        shade_familiar=False,
        title_suffix="",
        x_col="checkpoint_idx",
        xlabel="Communication round",
    )
    ff.save_figure(fig, slug, out_dir=FIG_DIR)
    plt.show()
""")

md("""\
### 6b Overall validation mAP per round (figs 10a-10b)
""")

code("""\
TRAJ_CURATED = [
    "fed_no_filter_cityday_curated",
    "fed_adaptive_window_p20_twoRef_cityday_curated",
    "fed_adaptive_reservoir_p20_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    "fed_random_p15_cityday_curated",
]
TRAJ_CURATED = [v for v in TRAJ_CURATED if v in inv["variant"].tolist()]
traj = fa.mAP_trajectory(TRAJ_CURATED, project_root=PROJECT_ROOT)
fig, _ = ff.plot_overall_mAP_trajectory(
    traj, x_col="round", xlabel="Communication round",
    smoothing_window=TRAJ_SMOOTHING,
    show_std_for=["fed_adaptive_reservoir_p20_twoRef_cityday_curated"],
    title="Default schedule, curated partition",
)
ff.save_figure(fig, "10a_overall_mAP_trajectory_default_curated", out_dir=FIG_DIR)
plt.show()
""")

code("""\
TRAJ_TMP = [
    "fed_no_filter_cityday_temporal",
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal",
    "fed_random_p19_cityday_temporal",
]
TRAJ_TMP = [v for v in TRAJ_TMP if v in inv["variant"].tolist()]
traj_tmp = fa.mAP_trajectory(TRAJ_TMP, project_root=PROJECT_ROOT)
fig, _ = ff.plot_overall_mAP_trajectory(
    traj_tmp, x_col="round", xlabel="Communication round",
    smoothing_window=TRAJ_SMOOTHING,
    show_std_for=["fed_adaptive_reservoir_p20_twoRef_cityday_temporal"],
    title="Default schedule, temporal partition",
)
ff.save_figure(fig, "10b_overall_mAP_trajectory_default_temporal", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 7. Schedule x partition mechanism (table only)
# =============================================================================

md("""\
## 7 Schedule x partition mechanism (table)
""")

code("""\
mech = fa.iso_accept_table([(f, r) for (_, f, r) in CELL_HEADLINE_PAIRS],
                           project_root=PROJECT_ROOT, tail_k=TAIL_K)
mech.insert(0, "cell", [lab for (lab, _, _) in CELL_HEADLINE_PAIRS])
mech.to_csv(TABLE_DIR / "schedule_x_partition.csv", index=False)
display(mech[["cell", "filter_accept", "random_accept",
              "filter_smoothed", "random_smoothed", "delta_smoothed"]].round(4))
""")

# =============================================================================
# 8. Ablations (figs 11-13)
# =============================================================================

md("""\
## 8 Methodology ablations (figs 11-13)
""")

md("### 8a Two-reference vs single-reference Mahalanobis (fig 11)")

code("""\
abl_twoRef = fa.ablation_pair_table(fa.ABLATION_PAIRINGS["twoRef"],
                                    project_root=PROJECT_ROOT, tail_k=TAIL_K)
abl_twoRef.to_csv(TABLE_DIR / "ablation_twoRef.csv", index=False)
display(abl_twoRef.round(4))
fig, _ = ff.plot_ablation_pair_bar(
    abl_twoRef,
    baseline_label="single-ref",
    ablated_label="two-ref",
    ymin=0.20,
)
ff.save_figure(fig, "11_ablation_twoRef", out_dir=FIG_DIR)
plt.show()
""")

md("### 8b Tighter accept budget (p20 vs p10) (fig 12)")

code("""\
abl_tight = fa.ablation_pair_table(fa.ABLATION_PAIRINGS["tighter_accept"],
                                   project_root=PROJECT_ROOT, tail_k=TAIL_K)
abl_tight.to_csv(TABLE_DIR / "ablation_tighter_accept.csv", index=False)
display(abl_tight.round(4))
fig, _ = ff.plot_ablation_pair_bar(
    abl_tight,
    baseline_label=r"$\\tau_{20}$ baseline",
    ablated_label=r"$\\tau_{10}$ / $\\tau_{15}$ tighter",
    ymin=0.20,
)
ff.save_figure(fig, "12_ablation_tighter_accept", out_dir=FIG_DIR)
plt.show()
""")

md("### 8c Default vs heavy-local schedule (fig 13)")

code("""\
abl_hl = fa.ablation_pair_table(fa.ABLATION_PAIRINGS["heavyLocal"],
                                project_root=PROJECT_ROOT, tail_k=TAIL_K)
abl_hl.to_csv(TABLE_DIR / "ablation_heavyLocal.csv", index=False)
display(abl_hl.round(4))
fig, _ = ff.plot_ablation_pair_bar(
    abl_hl,
    baseline_label="default schedule",
    ablated_label="heavy-local",
    ymin=0.20,
)
ff.save_figure(fig, "13_ablation_heavyLocal", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 9. Per-block summary tables
# =============================================================================

md("""\
## 9 Per-block summary tables
""")

code("""\
DEFAULT_CURATED = [v for v in fa.FEATURED_VARIANTS
                   if fa.manifest_for_variant(v) == "curated"
                   and fa.schedule_for_variant(v) == "default"]
grid = fa.per_domain_grid(DEFAULT_CURATED,
                          project_root=PROJECT_ROOT, tail_k=TAIL_K)
grid.to_csv(TABLE_DIR / "per_domain_curated.csv")
summary_cur = fa.per_domain_summary(DEFAULT_CURATED,
                                    project_root=PROJECT_ROOT, tail_k=TAIL_K)
summary_cur.to_csv(TABLE_DIR / "per_domain_summary_curated.csv", index=False)
display(summary_cur.round(4))
""")

code("""\
TEMPORAL = [v for v in fa.FEATURED_VARIANTS
            if fa.manifest_for_variant(v) == "temporal"]
grid_tmp = fa.per_domain_grid(TEMPORAL,
                              project_root=PROJECT_ROOT, tail_k=TAIL_K)
grid_tmp.to_csv(TABLE_DIR / "per_domain_temporal.csv")
summary_tmp = fa.per_domain_summary(TEMPORAL,
                                    project_root=PROJECT_ROOT, tail_k=TAIL_K)
summary_tmp.to_csv(TABLE_DIR / "per_domain_summary_temporal.csv", index=False)
display(summary_tmp.round(4))
""")

code("""\
HL_CURATED = [v for v in fa.FEATURED_VARIANTS
              if fa.manifest_for_variant(v) == "curated"
              and fa.schedule_for_variant(v) == "heavyLocal"]
grid_hl = fa.per_domain_grid(HL_CURATED,
                             project_root=PROJECT_ROOT, tail_k=TAIL_K)
grid_hl.to_csv(TABLE_DIR / "per_domain_heavyLocal.csv")
summary_hl = fa.per_domain_summary(HL_CURATED,
                                   project_root=PROJECT_ROOT, tail_k=TAIL_K)
summary_hl.to_csv(TABLE_DIR / "per_domain_summary_heavyLocal.csv", index=False)
display(summary_hl.round(4))
""")


# =============================================================================
# 10. Full sweeps (figs A1-A5)
# =============================================================================

md("""\
## 10 Full per-block sweeps
""")

md("""\
### 10a Per-stream-block trajectory facets (fig A1)
""")

code("""\
APPENDIX_TRAJ_HEADLINE = [
    "fed_no_filter_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    "fed_random_p15_cityday_curated",
]
ALL_BLOCKS = sorted(delta_grid.index.tolist())
trajectories = {
    v: fa.per_block_trajectory(v, ALL_BLOCKS, project_root=PROJECT_ROOT)
    for v in APPENDIX_TRAJ_HEADLINE
}
fig, _ = ff.plot_per_block_trajectory(
    trajectories, ALL_BLOCKS,
    n_cols=4,
    smoothing_window=TRAJ_SMOOTHING,
    show_std=True,
)
ff.save_figure(fig, "A1_per_block_trajectory_default_curated_full",
               out_dir=FIG_DIR)
plt.show()
""")

md("### 10b Full per-block heatmap -- default + curated (fig A2)")

code("""\
fig, _ = ff.plot_per_domain_heatmap(grid)
ff.save_figure(fig, "A2_per_domain_heatmap_default_curated_full", out_dir=FIG_DIR)
plt.show()
""")

md("### 10c Full per-block heatmap -- heavy-local + curated (fig A3)")

code("""\
fig, _ = ff.plot_per_domain_heatmap(grid_hl)
ff.save_figure(fig, "A3_per_domain_heatmap_heavyLocal_curated_full", out_dir=FIG_DIR)
plt.show()
""")

md("### 10d Full per-block heatmap -- temporal partition (fig A4)")

code("""\
fig, _ = ff.plot_per_domain_heatmap(grid_tmp)
ff.save_figure(fig, "A4_per_domain_heatmap_temporal_full", out_dir=FIG_DIR)
plt.show()
""")

md("### 10e Per-class breakdown (headline variants) (fig A5 / table)")

code("""\
PER_CLASS_VARIANTS = HEADLINE_VARIANTS_DEFAULT_CURATED
per_class = fa.per_class_grid(PER_CLASS_VARIANTS, project_root=PROJECT_ROOT, tail_k=TAIL_K)
per_class.to_csv(TABLE_DIR / "per_class_curated.csv")
display(per_class.round(3))
""")


# =============================================================================
# 11. Saved figures and tables index
# =============================================================================

md("## 11 Saved figures and tables index")

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
