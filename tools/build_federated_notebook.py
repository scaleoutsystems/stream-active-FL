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

Tables and figures backing the federated chapter.  Heavy lifting lives
in two package modules:

- `stream_active_fl.analysis.federated` — variant registry, headline
  tables, per-client routing, per-block grids, iso-accept pairings,
  ablation comparisons.
- `stream_active_fl.analysis.figures.federated` — accept-rate
  dynamics, per-block heatmaps, trajectories, ablation bars.

Outputs:

- Tables  -> `reports/federated/tables/*.csv`
- Figures -> `reports/federated/figures/*.{pdf,png}`

The chapter spans a 2 x 2 design: **schedule** (default 30 rounds x
1 000 items vs heavy-local 10 rounds x 3 000 items) crossed with
**partition** (curated domain-aligned vs temporal time-aligned).
The four cells produce qualitatively different results; tracing
when the filter beats random and when it ties is the chapter's
central mechanism story.

Storyline:

1. **Iso-accept overall mAP** — does the filter beat random *at the
   same accept budget*?  Reported one panel per cell.
2. **Per-client compute routing** — does the filter send more compute
   to novel-domain clients?  Random has flat per-client accept rates
   by construction.
3. **Per-block validation mAP** — does the routing translate to a
   per-domain gain on the validation set, or does FedAvg average it
   away?
4. **Schedule x partition mechanism table + refresh-density
   diagnostic** — when does the filter beat random and when does it
   tie?
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
from stream_active_fl.analysis import federated as fa
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

fa.prime_registry(project_root=PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")
print(f"Variants registered: {len(fa.FEATURED_VARIANTS)}")
""")

md("""\
### 1a Headline variants and pairs

The headline figures below use a small fixed variant set per cell.
The headline filter is **reservoir p20 twoRef** — the best-performing
variant on the default + curated cell — repeated across the four
cells.  Each cell has 2-3 random partners chosen so iso-accept
comparisons are tight (accept-rate gap < 0.025).""")

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

# Iso-accept pairs used for the headline scatter / delta heatmap
# in each cell of the 2 x 2 design.
HEADLINE_PAIRS_DEFAULT_CURATED = [
    ("fed_static_p20_cityday_curated",                   "fed_random_p77_cityday_curated"),
    ("fed_adaptive_window_p20_twoRef_cityday_curated",   "fed_random_p12_cityday_curated"),
    ("fed_adaptive_reservoir_p20_cityday_curated",       "fed_random_p18_cityday_curated"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_curated","fed_random_p15_cityday_curated"),
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
# (filter, random) pair.  Used by the schedule x partition table
# (Section 6).
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
# 3. Iso-accept comparison — per cell
# =============================================================================

md("""\
## 3 Iso-accept comparison

The fairest filter-vs-random test is *iso-accept*: each filter is
paired with a random baseline whose `accept_fraction` matches the
filter's empirical accept rate.  Positive Δ smoothed mAP means the
filter trains a better global model than random at the same per-frame
budget.

We report one scatter per cell of the 2 x 2 design rather than a
single combined panel: each cell has its own accept-rate range and
its own iso-accept partners, and combining them obscures the
mechanism story.""")

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

# Each per-cell scatter is restricted to that cell's HEADLINE pairs so
# the plot stays a clean 2-4 markers — the budget-sweep view lives in
# the appendix.
code("""\
def _iso_subset(iso_df, pairs):
    f_keys = {f for f, _ in pairs}
    return iso_df[iso_df["filter_variant"].isin(f_keys)]
""")

md("### 3a Default schedule + curated partition")

code("""\
iso_dc = _iso_subset(iso, HEADLINE_PAIRS_DEFAULT_CURATED)
fig, _ = ff.plot_iso_accept_scatter(iso_dc, manifest="curated", schedule="default")
ff.save_figure(fig, "01_iso_accept_default_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 3b Default schedule + temporal partition")

code("""\
iso_dt = _iso_subset(iso, HEADLINE_PAIRS_DEFAULT_TEMPORAL)
fig, _ = ff.plot_iso_accept_scatter(iso_dt, manifest="temporal", schedule="default")
ff.save_figure(fig, "02_iso_accept_default_temporal", out_dir=FIG_DIR)
plt.show()
""")

md("### 3c Heavy-local schedule + curated partition")

code("""\
iso_hc = _iso_subset(iso, HEADLINE_PAIRS_HL_CURATED)
fig, _ = ff.plot_iso_accept_scatter(iso_hc, manifest="curated", schedule="heavyLocal")
ff.save_figure(fig, "03_iso_accept_heavyLocal_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 3d Heavy-local schedule + temporal partition")

code("""\
iso_ht = _iso_subset(iso, HEADLINE_PAIRS_HL_TEMPORAL)
fig, _ = ff.plot_iso_accept_scatter(iso_ht, manifest="temporal", schedule="heavyLocal")
ff.save_figure(fig, "04_iso_accept_heavyLocal_temporal", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 3e Per-block Δ-mAP heatmap — schedule x partition

Four columns, one per cell of the 2 x 2 design, taken at each cell's
best filter (reservoir p20 twoRef) paired with its tightest iso-accept
random.  Familiar blocks (`city_day_clear`/`cloudy`) are pinned at the
top and separated from novel blocks by a horizontal line.""")

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
""")

code("""\
fig, _ = ff.plot_per_domain_delta_heatmap(delta_grid)
ff.save_figure(fig, "05_per_domain_delta_headline", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 3f Per-block trajectory Δ (training history)

The tail-`TAIL_K` heatmap captures end-of-training behavior; the
*cumulative* delta integrates over the whole training trajectory and
catches whether the filter held an advantage *throughout* training,
even if its tail values are tied.""")

code("""\
traj_delta = fa.per_block_trajectory_delta(CELL_PAIRS,
                                           project_root=PROJECT_ROOT)
traj_delta.to_csv(TABLE_DIR / "per_block_trajectory_delta.csv", index=False)
fig, _ = ff.plot_per_block_trajectory_delta(traj_delta, metric="cum_avg_delta")
ff.save_figure(fig, "06_per_block_trajectory_delta_headline", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 4. Per-client compute routing
# =============================================================================

md("""\
## 4 Per-client compute routing

This is the federation-specific test.  Random distributes accepts
uniformly across clients; an "intelligent" filter should over-accept
on clients whose data is novel relative to the bootstrap and
under-accept on the familiar one.  Two partition strategies are used:

- **Curated (`domain_aligned`)** — each client owns a coherent domain
  shard:
  - C0 = `city_day_clear/cloudy` (familiar — bootstrap mode)
  - C1 = `city_day_rain_wet/snow`, `city_twilight/night`
  - C2 = `arterial-urban_day/twi-night`
  - C3 = `highway_*`, `arterial-rural_*`, `smaller-rural_all`
- **Temporal (`contiguous`)** — each client owns one chronological
  quartile of the post-bootstrap stream; domain mixes are similar
  across clients but offset in time.""")

code("""\
per_client = fa.per_client_accept_table(project_root=PROJECT_ROOT)
per_client.to_csv(TABLE_DIR / "per_client_accept.csv", index=False)
display(per_client.head(20).round(4))
""")

md("### 4a Per-client accept rate — default schedule, curated partition")

code("""\
ROUTING_VARIANTS_DEFAULT = [
    "fed_static_p20_cityday_curated",
    "fed_adaptive_window_p20_twoRef_cityday_curated",
    "fed_adaptive_reservoir_p20_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    "fed_random_p15_cityday_curated",
    "fed_random_p18_cityday_curated",
]
fig, _ = ff.plot_per_client_accept_rates(
    per_client, variants=ROUTING_VARIANTS_DEFAULT)
ff.save_figure(fig, "07_per_client_accept_default_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 4b Per-client accept rate — heavy-local schedule, curated partition")

code("""\
ROUTING_VARIANTS_HL = [
    "fed_adaptive_window_p20_twoRef_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p15_twoRef_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
    "fed_random_p21_cityday_curated_heavyLocal",
    "fed_random_p26_cityday_curated_heavyLocal",
]
fig, _ = ff.plot_per_client_accept_rates(
    per_client, variants=ROUTING_VARIANTS_HL)
ff.save_figure(fig, "08_per_client_accept_heavyLocal_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 4c Per-client accept rate — temporal partition")

code("""\
ROUTING_VARIANTS_TMP = [
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal",
    "fed_adaptive_reservoir_p15_twoRef_cityday_temporal_heavyLocal",
    "fed_adaptive_reservoir_p20_twoRef_cityday_temporal_heavyLocal",
    "fed_random_p19_cityday_temporal",
    "fed_random_p25_cityday_temporal_heavyLocal",
]
fig, _ = ff.plot_per_client_accept_rates(
    per_client, variants=ROUTING_VARIANTS_TMP)
ff.save_figure(fig, "09_per_client_accept_temporal", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 4d Novelty routing leaderboard

`novelty_ratio = mean(C1, C2, C3 accept rate) / C0 accept rate`.
Bars > 1 mean the variant routes more compute to novel-domain clients
than to the familiar one; bars ~ 1 mean flat routing (random's
behavior by construction).""")

code("""\
novelty = fa.novelty_routing_summary(project_root=PROJECT_ROOT)
novelty.to_csv(TABLE_DIR / "novelty_routing.csv", index=False)
display(novelty.round(4))
fig, _ = ff.plot_novelty_routing(novelty, manifest="curated", schedule="default")
ff.save_figure(fig, "10_novelty_routing_default_curated", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 5. Per-block validation mAP — headline cell
# =============================================================================

md("""\
## 5 Per-block validation mAP — default + curated cell

Smoothed tail-`TAIL_K` mAP per `stream_block` bucket per variant on
the headline cell (default schedule, curated partition).  This is the
absolute view; Section 3e shows the iso-accept *delta* view, which is
harder to game.""")

code("""\
grid = fa.per_domain_grid(HEADLINE_VARIANTS_DEFAULT_CURATED,
                          project_root=PROJECT_ROOT, tail_k=TAIL_K)
grid.to_csv(TABLE_DIR / "per_domain_curated.csv")
display(grid.round(3))

summary = fa.per_domain_summary(HEADLINE_VARIANTS_DEFAULT_CURATED,
                                project_root=PROJECT_ROOT, tail_k=TAIL_K)
summary.to_csv(TABLE_DIR / "per_domain_summary_curated.csv", index=False)
display(summary.round(4))

fig, _ = ff.plot_per_domain_heatmap(
    grid)
ff.save_figure(fig, "11_per_domain_heatmap_default_curated", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 5a Per-block trajectory facets

One panel per validation block; lines per variant.  Familiar blocks
are shaded so the eye can quickly compare novel vs familiar
trajectories.""")

code("""\
TRAJ_HEADLINE = [
    "fed_no_filter_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    "fed_random_p15_cityday_curated",
    "fed_random_p18_cityday_curated",
]
ALL_BLOCKS = sorted(grid.index.tolist())
trajectories = {
    v: fa.per_block_trajectory(v, ALL_BLOCKS, project_root=PROJECT_ROOT)
    for v in TRAJ_HEADLINE
}
fig, _ = ff.plot_per_block_trajectory(
    trajectories, ALL_BLOCKS,
    n_cols=4)
ff.save_figure(fig, "12_per_block_trajectory_default_curated", out_dir=FIG_DIR)
plt.show()
""")

md("### 5b Overall val mAP per round (default + curated)")

code("""\
TRAJ_OVERALL = [
    "fed_no_filter_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    "fed_random_p15_cityday_curated",
    "fed_random_p77_cityday_curated",
]
TRAJ_OVERALL = [v for v in TRAJ_OVERALL if v in inv["variant"].tolist()]
traj = fa.mAP_trajectory(TRAJ_OVERALL, project_root=PROJECT_ROOT)
fig, _ = ff.plot_overall_mAP_trajectory(
    traj, x_col="round")
ff.save_figure(fig, "13_overall_mAP_trajectory_default_curated", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 6. Schedule x partition mechanism + refresh-density diagnostic
# =============================================================================

md("""\
## 6 Schedule x partition mechanism

The 2 x 2 design at the cell-headline pair (best filter in each cell
paired with its tightest iso-accept random).  The filter beats random
by ~ +0.005 mAP on default + curated and on heavy-local + temporal,
and ties or loses on the other two cells.  The asymmetry is best
explained by the **refresh-density** diagnostic below: refresh density
controls effective accept rate, which in turn controls iso-accept
performance.""")

code("""\
mech = fa.iso_accept_table([(f, r) for (_, f, r) in CELL_HEADLINE_PAIRS],
                           project_root=PROJECT_ROOT, tail_k=TAIL_K)
mech.insert(0, "cell", [lab for (lab, _, _) in CELL_HEADLINE_PAIRS])
mech.to_csv(TABLE_DIR / "schedule_x_partition.csv", index=False)
display(mech[["cell", "filter_accept", "random_accept",
              "filter_smoothed", "random_smoothed", "delta_smoothed"]].round(4))
""")

md("""\
### 6a Refresh-density diagnostic (sparse refresh)

`fed_adaptive_reservoir_p20_twoRef_sparseRefresh_cityday_curated`
holds the schedule fixed (30 rounds x 1 000 items, default) but refits
the scoring reference only **every 3 rounds** instead of every round.
This isolates "stale reference" from "intense per-round local
training": at default cadence the curated cell shows the headline
+0.005 mAP advantage; at sparse cadence it loses to random by
−0.001 mAP at iso-accept.  The empirical accept rate inflates from
0.155 to 0.259 — structurally the same accept rate as the
heavy-local + curated cell that fails.""")

code("""\
DIAGNOSTIC_PAIRS = [
    ("fed_adaptive_reservoir_p20_twoRef_cityday_curated",
     "fed_random_p15_cityday_curated"),
    ("fed_adaptive_reservoir_p20_twoRef_sparseRefresh_cityday_curated",
     "fed_random_p26_cityday_curated"),
    ("fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal",
     "fed_random_p26_cityday_curated_heavyLocal"),
]
diag = fa.iso_accept_table(DIAGNOSTIC_PAIRS,
                           project_root=PROJECT_ROOT, tail_k=TAIL_K)
diag.to_csv(TABLE_DIR / "iso_accept_refresh_density.csv", index=False)
display(diag[iso_cols].round(4))
""")


# =============================================================================
# 7. Ablations
# =============================================================================

md("""\
## 7 Methodology ablations

Three controlled comparisons.  Each pairs a baseline variant with an
ablated variant at the same schedule + partition cell so the
delta isolates the design choice.""")

md("### 7a Two-reference vs single-reference Mahalanobis")

code("""\
abl_twoRef = fa.ablation_pair_table(fa.ABLATION_PAIRINGS["twoRef"],
                                    project_root=PROJECT_ROOT, tail_k=TAIL_K)
abl_twoRef.to_csv(TABLE_DIR / "ablation_twoRef.csv", index=False)
display(abl_twoRef.round(4))
fig, _ = ff.plot_ablation_pair_bar(abl_twoRef)
ff.save_figure(fig, "14_ablation_twoRef", out_dir=FIG_DIR)
plt.show()
""")

md("### 7b Tighter accept budget (p20 vs p10)")

code("""\
abl_tight = fa.ablation_pair_table(fa.ABLATION_PAIRINGS["tighter_accept"],
                                   project_root=PROJECT_ROOT, tail_k=TAIL_K)
abl_tight.to_csv(TABLE_DIR / "ablation_tighter_accept.csv", index=False)
display(abl_tight.round(4))
fig, _ = ff.plot_ablation_pair_bar(abl_tight)
ff.save_figure(fig, "15_ablation_tighter_accept", out_dir=FIG_DIR)
plt.show()
""")

md("### 7c Default vs heavy-local schedule")

code("""\
abl_hl = fa.ablation_pair_table(fa.ABLATION_PAIRINGS["heavyLocal"],
                                project_root=PROJECT_ROOT, tail_k=TAIL_K)
abl_hl.to_csv(TABLE_DIR / "ablation_heavyLocal.csv", index=False)
display(abl_hl.round(4))
fig, _ = ff.plot_ablation_pair_bar(abl_hl)
ff.save_figure(fig, "16_ablation_heavyLocal", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 8. Appendix — full sweeps
# =============================================================================

md("""\
## 8 Appendix — full variant sweeps

Full per-block heatmaps for each cell of the 2 x 2 design.  These do
not feed the main narrative; they are included for completeness.""")

md("### 8a Full per-block heatmap — default + curated")

code("""\
DEFAULT_CURATED = [v for v in fa.FEATURED_VARIANTS
                   if fa.manifest_for_variant(v) == "curated"
                   and fa.schedule_for_variant(v) == "default"]
grid_full = fa.per_domain_grid(DEFAULT_CURATED,
                               project_root=PROJECT_ROOT, tail_k=TAIL_K)
fig, _ = ff.plot_per_domain_heatmap(
    grid_full)
ff.save_figure(fig, "A1_per_domain_heatmap_default_curated_full", out_dir=FIG_DIR)
plt.show()
""")

md("### 8b Full per-block heatmap — heavy-local + curated")

code("""\
HL_CURATED = [v for v in fa.FEATURED_VARIANTS
              if fa.manifest_for_variant(v) == "curated"
              and fa.schedule_for_variant(v) == "heavyLocal"]
grid_hl = fa.per_domain_grid(HL_CURATED,
                             project_root=PROJECT_ROOT, tail_k=TAIL_K)
fig, _ = ff.plot_per_domain_heatmap(
    grid_hl)
ff.save_figure(fig, "A2_per_domain_heatmap_heavyLocal_curated_full", out_dir=FIG_DIR)
plt.show()
""")

md("### 8c Full per-block heatmap — temporal partition")

code("""\
TEMPORAL = [v for v in fa.FEATURED_VARIANTS
            if fa.manifest_for_variant(v) == "temporal"]
grid_tmp = fa.per_domain_grid(TEMPORAL,
                              project_root=PROJECT_ROOT, tail_k=TAIL_K)
grid_tmp.to_csv(TABLE_DIR / "per_domain_temporal.csv")
fig, _ = ff.plot_per_domain_heatmap(
    grid_tmp)
ff.save_figure(fig, "A3_per_domain_heatmap_temporal_full", out_dir=FIG_DIR)
plt.show()
""")

md("### 8d Per-class breakdown (headline variants)")

code("""\
PER_CLASS_VARIANTS = HEADLINE_VARIANTS_DEFAULT_CURATED
per_class = fa.per_class_grid(PER_CLASS_VARIANTS, project_root=PROJECT_ROOT, tail_k=TAIL_K)
per_class.to_csv(TABLE_DIR / "per_class_curated.csv")
display(per_class.round(3))
""")


# =============================================================================
# 9. Saved figures and tables index
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
