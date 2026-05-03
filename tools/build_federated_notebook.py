"""Generate notebooks/02_federated_analysis.ipynb from a structured cell list.

Run as ``python tools/build_federated_notebook.py`` from the project
root.  Add ``--execute`` to also run the resulting notebook end-to-end
so the cells contain their outputs (~2 minutes wall clock).

Re-runs without ``--execute`` overwrite cell *source* but produce a
notebook with empty outputs.  Use with care: if the notebook has been
executed and you only need to refresh structure, prefer editing the
cell list and re-executing afterwards.

The federated notebook mirrors the streaming one
(`01_streaming_analysis.ipynb`): heavy lifting lives in
`stream_active_fl.analysis.federated` and
`stream_active_fl.analysis.figures.federated`; cells stay thin so the
analysis logic is testable and reusable.
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

This notebook produces every table and figure that backs the federated
chapter of the write-up.  Heavy lifting lives in two reusable modules:

- `federated_analysis.py` — variant registry, headline tables, per-client
  routing, per-block grids, iso-accept pairings, ablation comparisons.
- `federated_figures.py` — accept-rate dynamics, per-block heatmaps,
  trajectories, ablation bars.

Outputs:

- Tables  -> `reports/federated/tables/*.csv`
- Figures -> `reports/federated/figures/*.{pdf,png}`

Storyline (mirroring streaming, but with federation-specific nuances):

1. **Iso-accept overall mAP** — does the filter beat random *at the same
   accept budget*?  This is the headline test for any active-learning
   filter.
2. **Per-client compute routing** — does the filter send more compute
   to novel-domain clients than to the familiar one?  Federated-only
   diagnostic: random has flat per-client accept rates by construction.
3. **Per-stream_block mAP** — does the routing translate to a per-block
   gain on the validation set, or does FedAvg average it away?
4. **Schedule and budget ablations** — does heavier local training or a
   tighter accept budget recover any per-novel-block gain we don't
   see at the default settings?

Re-run end to end after a new batch of experiments; everything is keyed
off `outputs/federated/<variant>/seed_<N>/<timestamp>/`.
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

#: Tail-k for smoothed mAP.  5 matches the streaming chapter.
TAIL_K = 5

fa.prime_registry(project_root=PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")
print(f"Figures -> {FIG_DIR}")
print(f"Tables  -> {TABLE_DIR}")
print(f"Variants registered: {len(fa.FEATURED_VARIANTS)}")
""")


# =============================================================================
# 2. Inventory
# =============================================================================

md("""\
## 2 Variant inventory

One row per variant with empirical accept rate, smoothed tail-`TAIL_K`
mAP across 3 seeds, total items processed and total optimizer steps,
plus a `schedule` column distinguishing default (30 rounds × 1 000
items) from `heavyLocal` (10 rounds × 3 000 items).""")

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
# 3. Iso-accept story --- headline
# =============================================================================

md("""\
## 3 Iso-accept comparison — headline

The fairest filter-vs-random test is *iso-accept*: each filter is paired
with a random baseline whose `accept_fraction` is set to match the
filter's empirical accept rate.  Positive Δ smoothed mAP means the
filter trains a better global model than random at the *same* per-frame
budget.""")

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

code("""\
fig, _ = ff.plot_iso_accept_scatter(iso, manifest="curated", schedule="default")
ff.save_figure(fig, "01_iso_accept_curated_default", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 3.1 Per-block iso-accept Δ-mAP (heatmap)

For each filter ↔ random pair we compute the smoothed-tail-`TAIL_K`
per-`stream_block` mAP and report the cell-wise Δ.  Familiar blocks
(`city_day_clear`/`cloudy`) are pinned at the top and separated from
novel blocks by a horizontal line.""")

code("""\
delta_grid = fa.per_domain_delta_grid(project_root=PROJECT_ROOT, tail_k=TAIL_K)
delta_grid.to_csv(TABLE_DIR / "per_domain_delta_curated.csv")
display(delta_grid.round(4))

delta_summary = fa.per_domain_delta_summary(project_root=PROJECT_ROOT, tail_k=TAIL_K)
delta_summary.to_csv(TABLE_DIR / "per_domain_delta_summary.csv", index=False)
display(delta_summary.round(4))
""")

code("""\
fig, _ = ff.plot_per_domain_delta_heatmap(delta_grid)
ff.save_figure(fig, "02_per_domain_delta_curated", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 3.2 Per-block *trajectory* Δ (training history)

The tail-`TAIL_K` heatmap captures end-of-training behavior only; the
*cumulative* delta integrates over the whole training trajectory and
catches whether the filter held an advantage *throughout* training, even
if its tail values are tied.  For each pair and block we report the
mean over rounds of `(filter mAP - random mAP)`.""")

code("""\
traj_delta = fa.per_block_trajectory_delta(project_root=PROJECT_ROOT)
traj_delta.to_csv(TABLE_DIR / "per_block_trajectory_delta.csv", index=False)
fig, _ = ff.plot_per_block_trajectory_delta(traj_delta, metric="cum_avg_delta")
ff.save_figure(fig, "03_per_block_trajectory_delta_curated", out_dir=FIG_DIR)
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
under-accept on the familiar one.  Clients are partitioned by
`domain_aligned`:

- C0 = `city_day_clear/cloudy`  *(familiar — bootstrap mode)*
- C1 = `city_day_rain_wet/snow`, `city_twilight/night`
- C2 = `arterial-urban_day/twi-night`
- C3 = `highway_*`, `arterial-rural_*`, `smaller-rural_all`""")

code("""\
per_client = fa.per_client_accept_table(project_root=PROJECT_ROOT)
per_client.to_csv(TABLE_DIR / "per_client_accept.csv", index=False)
display(per_client.head(20).round(4))
""")

code("""\
ROUTING_VARIANTS_DEFAULT = [
    "fed_static_p20_cityday_curated",
    "fed_adaptive_window_p20_cityday_curated",
    "fed_adaptive_window_p20_twoRef_cityday_curated",
    "fed_adaptive_reservoir_p20_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    "fed_random_p15_cityday_curated",
    "fed_random_p18_cityday_curated",
    "fed_random_p77_cityday_curated",
]
fig, _ = ff.plot_per_client_accept_rates(per_client,
                                         variants=ROUTING_VARIANTS_DEFAULT,
                                         title="Per-client accept rate (curated, default schedule)")
ff.save_figure(fig, "04_per_client_accept_default", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 4.1 Novelty routing leaderboard

`novelty_ratio = mean(C1, C2, C3 accept rate) / C0 accept rate`.  >1 means
the variant routes more compute to novel-domain clients than to the
familiar one; ~1 (the dashed reference line) means flat routing
(random's behavior by construction).""")

code("""\
novelty = fa.novelty_routing_summary(project_root=PROJECT_ROOT)
novelty.to_csv(TABLE_DIR / "novelty_routing.csv", index=False)
display(novelty.round(4))

fig, _ = ff.plot_novelty_routing(novelty, manifest="curated", schedule="default")
ff.save_figure(fig, "05_novelty_routing_default", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 5. Per-block validation mAP
# =============================================================================

md("""\
## 5 Per-stream_block mAP

Smoothed tail-`TAIL_K` mAP per `stream_block` bucket per variant.  This
is the absolute view; `Section 3.1` shows the iso-accept *delta* view
which is harder to game.""")

code("""\
PER_BLOCK_VARIANTS = [v for v in fa.FEATURED_VARIANTS
                      if fa.manifest_for_variant(v) == "curated"
                      and fa.schedule_for_variant(v) == "default"]
grid = fa.per_domain_grid(PER_BLOCK_VARIANTS, project_root=PROJECT_ROOT, tail_k=TAIL_K)
grid.to_csv(TABLE_DIR / "per_domain_curated.csv")
display(grid.round(3))

summary = fa.per_domain_summary(PER_BLOCK_VARIANTS,
                                project_root=PROJECT_ROOT, tail_k=TAIL_K)
summary.to_csv(TABLE_DIR / "per_domain_summary_curated.csv", index=False)
display(summary.round(4))

fig, _ = ff.plot_per_domain_heatmap(grid, title="Per-block end-of-training mAP — curated, default")
ff.save_figure(fig, "06_per_domain_heatmap_curated_default", out_dir=FIG_DIR)
plt.show()
""")

md("""\
### 5.1 Per-block trajectory facets

One panel per validation block; lines per variant.  Familiar blocks are
shaded so the eye can quickly compare novel vs familiar trajectories.""")

code("""\
TRAJ_HEADLINE = [
    "fed_no_filter_cityday_curated",
    "fed_adaptive_reservoir_p20_cityday_curated",
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
    n_cols=4,
    title="Per-block mAP through training — reservoir p20 (twoRef) vs iso-accept randoms")
ff.save_figure(fig, "07_per_block_trajectory_curated", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 6. Overall mAP and per-class AP trajectories
# =============================================================================

md("""\
## 6 Overall mAP and per-class AP

Total-val mAP and per-class AP through training.  Used as supporting
evidence rather than the headline; the iso-accept and per-block stories
are the primary diagnostics.""")

code("""\
HEADLINE = [
    "fed_no_filter_cityday_curated",
    "fed_static_p20_cityday_curated",
    "fed_adaptive_reservoir_p20_cityday_curated",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated",
    "fed_random_p15_cityday_curated",
    "fed_random_p18_cityday_curated",
    "fed_random_p77_cityday_curated",
]
HEADLINE = [v for v in HEADLINE if v in inv["variant"].tolist()]
traj = fa.mAP_trajectory(HEADLINE, project_root=PROJECT_ROOT)
fig, _ = ff.plot_overall_mAP_trajectory(traj, x_col="round",
                                        title="Overall val mAP per round — curated, default schedule")
ff.save_figure(fig, "08_overall_mAP_trajectory_default", out_dir=FIG_DIR)
plt.show()
""")

code("""\
PER_CLASS_VARIANTS = HEADLINE
per_class = fa.per_class_grid(PER_CLASS_VARIANTS, project_root=PROJECT_ROOT, tail_k=TAIL_K)
per_class.to_csv(TABLE_DIR / "per_class_curated.csv")
display(per_class.round(3))
""")


# =============================================================================
# 7. Ablations
# =============================================================================

md("""\
## 7 Ablations

Headline ablations: twoRef vs single-reference, p20 vs p10, default vs
heavyLocal schedule, and static vs adaptive at p20.""")

code("""\
abl_twoRef = fa.ablation_pair_table(fa.ABLATION_PAIRINGS["twoRef"],
                                    project_root=PROJECT_ROOT, tail_k=TAIL_K)
abl_twoRef.to_csv(TABLE_DIR / "ablation_twoRef.csv", index=False)
display(abl_twoRef.round(4))
fig, _ = ff.plot_ablation_pair_bar(abl_twoRef,
                                   title="twoRef ablation: single-ref vs two-ref Mahalanobis")
ff.save_figure(fig, "09_ablation_twoRef", out_dir=FIG_DIR)
plt.show()
""")

code("""\
abl_tight = fa.ablation_pair_table(fa.ABLATION_PAIRINGS["tighter_accept"],
                                   project_root=PROJECT_ROOT, tail_k=TAIL_K)
abl_tight.to_csv(TABLE_DIR / "ablation_tighter_accept.csv", index=False)
display(abl_tight.round(4))
fig, _ = ff.plot_ablation_pair_bar(abl_tight,
                                   title="Tighter-accept ablation: p20 vs p10")
ff.save_figure(fig, "10_ablation_tighter_accept", out_dir=FIG_DIR)
plt.show()
""")

code("""\
abl_hl = fa.ablation_pair_table(fa.ABLATION_PAIRINGS["heavyLocal"],
                                project_root=PROJECT_ROOT, tail_k=TAIL_K)
abl_hl.to_csv(TABLE_DIR / "ablation_heavyLocal.csv", index=False)
display(abl_hl.round(4))
fig, _ = ff.plot_ablation_pair_bar(abl_hl,
                                   title="Heavier local training: 30x1000 vs 10x3000")
ff.save_figure(fig, "11_ablation_heavyLocal", out_dir=FIG_DIR)
plt.show()
""")

code("""\
abl_sva = fa.ablation_pair_table(fa.ABLATION_PAIRINGS["static_vs_adaptive"],
                                 project_root=PROJECT_ROOT, tail_k=TAIL_K)
abl_sva.to_csv(TABLE_DIR / "ablation_static_vs_adaptive.csv", index=False)
display(abl_sva.round(4))
""")


# =============================================================================
# 8. Refresh dynamics (adaptive only)
# =============================================================================

md("""\
## 8 Refresh dynamics

Inter-refresh accept-rate decay for adaptive filters (single seed
diagnostic).  Reservoir-style filters refill their reference between
refreshes; the accept rate within each segment shows whether the
reference shift induces a saw-tooth pattern.""")

code("""\
ADAPTIVE_VARIANTS = [v for v in fa.FEATURED_VARIANTS
                     if fa.family_for_variant(v) in {"window", "reservoir"}
                     and fa.schedule_for_variant(v) == "default"]
segments = fa.refresh_segment_table(ADAPTIVE_VARIANTS, project_root=PROJECT_ROOT, seed=42)
segments.to_csv(TABLE_DIR / "refresh_segments_default.csv", index=False)
display(segments.head(20).round(4))
fig, _ = ff.plot_refresh_segment_accept(
    segments, title="Inter-refresh accept rate (seed 42, default schedule)")
ff.save_figure(fig, "12_refresh_segments_default", out_dir=FIG_DIR)
plt.show()
""")


# =============================================================================
# 9. heavyLocal schedule: does heavier local training change the picture?
# =============================================================================

md("""\
## 9 heavyLocal schedule (Phase Z)

Phase Z replaces the default `30 rounds × 1 000 items` schedule with
`10 rounds × 3 000 items` (same total compute, 3× fewer aggregations).
The hypothesis is that less FedAvg dilution should let novel-block
specialization survive.  This panel mirrors Section 3 for the
heavyLocal variants.""")

code("""\
HL_VARIANTS = [v for v in fa.FEATURED_VARIANTS
               if fa.schedule_for_variant(v) == "heavyLocal"]
if HL_VARIANTS:
    hl_inv = inv[inv["schedule"] == "heavyLocal"]
    display(hl_inv[display_cols].round(4))

    iso_hl = iso[iso["schedule"] == "heavyLocal"]
    if not iso_hl.empty:
        display(iso_hl[iso_cols].round(4))
        fig, _ = ff.plot_iso_accept_scatter(iso, manifest="curated",
                                            schedule="heavyLocal")
        ff.save_figure(fig, "13_iso_accept_curated_heavyLocal", out_dir=FIG_DIR)
        plt.show()

    fig, _ = ff.plot_per_client_accept_rates(per_client, variants=HL_VARIANTS,
                                             title="Per-client accept rate — heavyLocal")
    ff.save_figure(fig, "14_per_client_accept_heavyLocal", out_dir=FIG_DIR)
    plt.show()

    fig, _ = ff.plot_novelty_routing(novelty, manifest="curated",
                                     schedule="heavyLocal")
    ff.save_figure(fig, "15_novelty_routing_heavyLocal", out_dir=FIG_DIR)
    plt.show()
else:
    print("No heavyLocal runs registered yet.")
""")


# =============================================================================
# 10. Saved figures and tables index
# =============================================================================

md("## 10 Saved figures and tables index")

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
