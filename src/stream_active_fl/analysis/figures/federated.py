"""
Figure generators for the federated experiments.

Mirrors the streaming figure module for the federated pipeline.  All
plotting calls return ``(fig, ax)`` (or ``(fig, axes)``) so the notebook
can save and inline them without baking the file system path into the
helper.  Use `save_figure` (re-exported from
`stream_active_fl.analysis.figures`) for the canonical PDF + PNG export.

Public surface:

    Style helpers
        FAMILY_COLORS                 -- variant-family palette
        VARIANT_COLOR_PALETTE         -- per-variant color overrides
        CLIENT_COLORS                 -- per-client palette
        variant_color(variant)        -- stable color from family
        variant_linestyle(variant)    -- solid / dashed / dotted from flavor
        variant_marker(variant)       -- marker shape from flavor
        save_figure(fig, name, fmt)   -- writes PDF + PNG into out_dir

    Headline figures
        plot_inventory_scatter(inv)
        plot_iso_accept_scatter(iso)
        plot_overall_mAP_trajectory(traj)
        plot_smoothed_leaderboard(inv)

    Per-client figures (federated-specific)
        plot_per_client_accept_rates(per_client_long, variants)
        plot_novelty_routing(novelty_df)

    Per-block figures
        plot_per_domain_heatmap(grid, *, vmin, vmax, title)
        plot_per_domain_delta_heatmap(delta_grid, ...)
        plot_per_domain_bars(grid, variants_to_show)
        plot_per_block_trajectory(trajectories_by_variant, blocks)
        plot_per_block_trajectory_delta(traj_delta_long)
        plot_balanced_vs_worst(summary_df)

    Ablations
        plot_ablation_pair_bar(pair_df, *, title)

    Refresh dynamics
        plot_refresh_segment_accept(segments_df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .. import runs as ah
from .. import federated as fa
from . import heatmap as _heatmap, save_figure  # noqa: F401  (re-export)


# =============================================================================
# Style
# =============================================================================

FAMILY_COLORS: Dict[str, str] = dict(ah.FILTER_FAMILY_COLORS)
FAMILY_COLORS["random"] = "#7f7f7f"

# Per-variant color overrides.  Distinct hues required for the
# trajectory plots whose +/-1 std bands overlap; variants are still
# loosely grouped by family hue so the legend reads naturally.
VARIANT_COLOR_PALETTE: Dict[str, str] = {
    # Phase 1A baselines
    "fed_no_filter_cityday_curated":                              "#2ca02c",  # green
    "fed_no_filter_cityday_curated_heavyLocal":                   "#1b5e1b",  # dark green
    # static — distinct purple
    "fed_static_p20_cityday_curated":                             "#9467bd",
    # window family — orange vs olive
    "fed_adaptive_window_p20_cityday_curated":                    "#ff7f0e",
    "fed_adaptive_window_p20_twoRef_cityday_curated":             "#bcbd22",
    "fed_adaptive_window_p10_cityday_curated":                    "#ffbb78",
    # reservoir family — red vs pink
    "fed_adaptive_reservoir_p20_cityday_curated":                 "#d62728",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated":          "#e377c2",
    "fed_adaptive_reservoir_p10_cityday_curated":                 "#fc8d62",
    "fed_adaptive_reservoir_p10_twoRef_cityday_curated":          "#f7b6d2",
    "fed_adaptive_reservoir_p20_cityday_curated_heavyLocal":      "#8b0000",
    "fed_adaptive_reservoir_p20_twoRef_cityday_curated_heavyLocal":
                                                                  "#c2185b",
    # random family — shades of grey
    "fed_random_p12_cityday_curated":                             "#cccccc",
    "fed_random_p15_cityday_curated":                             "#a0a0a0",
    "fed_random_p18_cityday_curated":                             "#808080",
    "fed_random_p77_cityday_curated":                             "#202020",
    "fed_random_p15_cityday_curated_heavyLocal":                  "#5b5b5b",
    "fed_random_p18_cityday_curated_heavyLocal":                  "#3b3b3b",
}


# Per-client palette for the federated 4-client `domain_aligned` setup.
CLIENT_COLORS: Dict[int, str] = {
    0: "#1f77b4",   # familiar - blue
    1: "#2ca02c",   # city-novel - green
    2: "#ff7f0e",   # urban arterial - orange
    3: "#d62728",   # out-of-city - red
}


def variant_color(variant: str, *, project_root: Optional[Path] = None) -> str:
    """Return a stable matplotlib color for a variant.

    Each *featured* variant in `VARIANT_COLOR_PALETTE` has a hand-picked
    distinct hue; everything else falls back to the family-level color
    so unfamiliar variants still render plausibly.
    """
    if variant in VARIANT_COLOR_PALETTE:
        return VARIANT_COLOR_PALETTE[variant]
    fam = fa.family_for_variant(variant, project_root=project_root)
    return FAMILY_COLORS.get(fam, "#444444")


def variant_linestyle(variant: str) -> str:
    """Return a stable matplotlib linestyle from variant flavor."""
    if "twoRef" in variant:
        return "--"
    if "noBoot" in variant:
        return ":"
    if "_p10_" in variant or "_p15_" in variant:
        return "--"
    if variant.endswith("_heavyLocal"):
        return "-."
    return "-"


def variant_marker(variant: str) -> str:
    """Return a stable marker shape from variant flavor."""
    if "twoRef" in variant:
        return "s"
    if "noBoot" in variant:
        return "X"
    if "_static_" in variant:
        return "D"
    if "_random_" in variant:
        return "."
    if "no_filter" in variant:
        return "*"
    return "o"


# =============================================================================
# Headline figures
# =============================================================================

def plot_inventory_scatter(
    inv: pd.DataFrame,
    *,
    manifest: str = "curated",
    schedule: str = "default",
    figsize: Tuple[float, float] = (7.5, 5.0),
) -> Tuple[Figure, Axes]:
    """Scatter of accept_rate (x) vs smoothed_mAP (y) with random envelope.

    Random baselines for the same (manifest, schedule) pair are joined
    into a reference curve so the eye can read "where would a random
    filter at this accept rate land".  Filters above the curve outperform
    iso-accept random.
    """
    sub = inv[(inv["manifest"] == manifest)
              & (inv["schedule"] == schedule)].copy()
    sub = sub.dropna(subset=["accept_rate", "smoothed_mAP"])
    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)

    rand = sub[sub["family"] == "random"].sort_values("accept_rate")
    if not rand.empty:
        ax.plot(rand["accept_rate"], rand["smoothed_mAP"],
                color=FAMILY_COLORS.get("random", "#7f7f7f"),
                linestyle="--", linewidth=1.2, alpha=0.8,
                label="random envelope", zorder=1)

    for _, row in sub.iterrows():
        std = row["smoothed_std"] if pd.notna(row["smoothed_std"]) else 0
        ax.errorbar(
            row["accept_rate"], row["smoothed_mAP"],
            yerr=std,
            fmt=variant_marker(row["variant"]),
            color=variant_color(row["variant"]),
            markersize=8, markeredgecolor="white", markeredgewidth=0.6,
            capsize=2, elinewidth=0.6, alpha=0.95, zorder=3,
        )
        ax.annotate(row["label"], (row["accept_rate"], row["smoothed_mAP"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color="#333333")

    ax.set_xlabel("Effective accept rate")
    ax.set_ylabel("Smoothed tail-5 mAP")
    sched_tag = f", {schedule}" if schedule != "default" else ""
    ax.set_title(f"Iso-accept leaderboard — {manifest}{sched_tag}",
                 fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    fams_in_plot = sorted(sub["family"].unique())
    handles = [mpatches.Patch(color=FAMILY_COLORS.get(f, "#444"), label=f)
               for f in fams_in_plot]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


def plot_iso_accept_scatter(
    iso: pd.DataFrame,
    *,
    manifest: str = "curated",
    schedule: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 5.0),
) -> Tuple[Figure, Axes]:
    """Plot Δsmoothed mAP vs filter accept_rate for each (filter, random) pair."""
    sub = iso[iso["manifest"] == manifest].copy()
    if schedule is not None and "schedule" in sub.columns:
        sub = sub[sub["schedule"] == schedule]
    fig, ax = plt.subplots(figsize=figsize)
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    for _, row in sub.iterrows():
        f_std = row.get("filter_smoothed_std", 0) or 0
        r_std = row.get("random_smoothed_std", 0) or 0
        ax.errorbar(
            row["filter_accept"], row["delta_smoothed"],
            yerr=np.sqrt(f_std ** 2 + r_std ** 2),
            fmt=variant_marker(row["filter_variant"]),
            color=variant_color(row["filter_variant"]),
            markersize=9, markeredgecolor="white", markeredgewidth=0.6,
            capsize=2, elinewidth=0.6, alpha=0.95,
        )
        ax.annotate(row["filter_label"],
                    (row["filter_accept"], row["delta_smoothed"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=7.5)

    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Filter accept rate")
    ax.set_ylabel(r"$\Delta$ smoothed mAP (filter $-$ iso-accept random)")
    sched_tag = f" ({schedule})" if schedule and schedule != "default" else ""
    ax.set_title(f"Iso-accept gain — {manifest}{sched_tag}", fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_overall_mAP_trajectory(
    traj: pd.DataFrame,
    *,
    x_col: str = "items_processed_total",
    figsize: Tuple[float, float] = (8.0, 4.5),
    title: str = "Federated mAP per round",
) -> Tuple[Figure, Axes]:
    """Plot overall mAP vs ``x_col`` for multiple variants (already aggregated).

    ``traj`` must come from `federated.mAP_trajectory`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if traj.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    if x_col not in traj.columns:
        x_col = "round"
    for v, grp in traj.groupby("variant", sort=False):
        grp = grp.sort_values(x_col)
        c = variant_color(v)
        ls = variant_linestyle(v)
        ax.plot(grp[x_col], grp["mAP"], color=c, linestyle=ls,
                label=fa.label_for(v), linewidth=1.4, alpha=0.9)
        if "mAP_std" in grp.columns:
            ax.fill_between(
                grp[x_col],
                grp["mAP"] - grp["mAP_std"].fillna(0),
                grp["mAP"] + grp["mAP_std"].fillna(0),
                color=c, alpha=0.12, linewidth=0,
            )

    ax.set_xlabel({"items_processed_total": "items processed (cumulative)",
                   "optimizer_steps_total": "optimizer steps (cumulative)",
                   "round": "round"}.get(x_col, x_col))
    ax.set_ylabel("mAP")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.85)
    fig.tight_layout()
    return fig, ax


def plot_smoothed_leaderboard(
    inv: pd.DataFrame,
    *,
    manifest: str = "curated",
    schedule: str = "default",
    sort_by: str = "smoothed_mAP",
    figsize: Tuple[float, float] = (8.0, 5.0),
) -> Tuple[Figure, Axes]:
    """Horizontal bar chart of smoothed-tail mAP, sorted descending."""
    sub = inv[(inv["manifest"] == manifest)
              & (inv["schedule"] == schedule)].copy()
    sub = sub.dropna(subset=[sort_by])
    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    sub = sub.sort_values(sort_by, ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    colors = [variant_color(v) for v in sub["variant"]]
    bars = ax.barh(sub["label"], sub[sort_by], color=colors,
                   xerr=sub.get("smoothed_std"), capsize=2,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel(sort_by.replace("_", " "))
    sched_tag = f" ({schedule})" if schedule != "default" else ""
    ax.set_title(f"{sort_by.replace('_', ' ')} — {manifest}{sched_tag}",
                 fontsize=11, loc="left")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, ar in zip(bars, sub["accept_rate"]):
        ax.text(bar.get_width() + 0.0008, bar.get_y() + bar.get_height() / 2,
                f"acc={ar:.2f}", va="center", fontsize=7, color="#333333")
    fig.tight_layout()
    return fig, ax


# =============================================================================
# Per-client figures (federated-specific)
# =============================================================================

def plot_per_client_accept_rates(
    per_client: pd.DataFrame,
    *,
    variants: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Per-client accept rate",
) -> Tuple[Figure, Axes]:
    """Grouped bar chart: one group per variant, one bar per client.

    Filters that *route* compute show stratified bars (C0 < C1, C2, C3
    typically).  Random variants show flat bars.
    """
    df = per_client.copy()
    if variants is not None:
        df = df[df["variant"].isin(list(variants))]
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize or (8.0, 4.5))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    variant_order = (list(variants) if variants is not None
                     else df["variant"].drop_duplicates().tolist())
    n_var = len(variant_order)
    n_clients = int(df["client"].max()) + 1
    if figsize is None:
        figsize = (max(8.0, 0.7 * n_var + 2.0), 4.5)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_var)
    bar_w = 0.85 / max(1, n_clients)
    for cid in range(n_clients):
        sub = df[df["client"] == cid].set_index("variant").reindex(variant_order)
        ax.bar(
            x + cid * bar_w - 0.425 + bar_w / 2,
            sub["accept_rate"].values,
            yerr=sub["accept_rate_std"].fillna(0).values,
            width=bar_w,
            color=CLIENT_COLORS.get(cid, "#888888"),
            label=fa.CLIENT_LABEL.get(cid, f"C{cid}"),
            edgecolor="white", linewidth=0.4,
            capsize=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([fa.label_for(v) for v in variant_order],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("accept rate")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=n_clients, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig, ax


def plot_novelty_routing(
    novelty: pd.DataFrame,
    *,
    manifest: str = "curated",
    schedule: str = "default",
    figsize: Tuple[float, float] = (7.5, 4.5),
    title: str = "Novelty-routing ratio",
) -> Tuple[Figure, Axes]:
    """Horizontal bar chart of novelty_ratio per variant.

    `novelty` is the output of
    `federated.novelty_routing_summary`.  Bars > 1 mean
    the variant routes more compute to novel-domain clients than to the
    familiar one; bars ~= 1 mean flat (random-like).
    """
    if "schedule" in novelty.columns:
        sub = novelty[novelty["schedule"] == schedule].copy()
    else:
        sub = novelty.copy()
    sub = sub[sub["variant"].apply(fa.manifest_for_variant) == manifest]
    sub = sub.dropna(subset=["novelty_ratio"])
    fig, ax = plt.subplots(figsize=figsize)
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    sub = sub.sort_values("novelty_ratio")
    colors = [variant_color(v) for v in sub["variant"]]
    ax.barh(sub["label"], sub["novelty_ratio"], color=colors,
            edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(r"novelty ratio = mean(C1, C2, C3 accept) / C0 accept")
    sched_tag = f" ({schedule})" if schedule != "default" else ""
    ax.set_title(f"{title} — {manifest}{sched_tag}", fontsize=11, loc="left")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax


# =============================================================================
# Per-block / per-domain figures
# =============================================================================

def plot_per_domain_heatmap(
    grid: pd.DataFrame,
    *,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Per-block end-of-training mAP",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Axes]:
    """Heatmap of (block x variant) mAP grid."""
    if figsize is None:
        figsize = (max(6.0, 0.55 * grid.shape[1] + 1.5),
                   max(3.5, 0.30 * grid.shape[0] + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    if grid.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    im = _heatmap(grid, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, title=title,
                  annotate=True, fmt="{:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mAP")
    fig.tight_layout()
    return fig, ax


def plot_per_domain_delta_heatmap(
    delta_grid: pd.DataFrame,
    *,
    vlim: Optional[float] = None,
    title: str = r"Per-block $\Delta$mAP (filter $-$ iso-accept random)",
    figsize: Optional[Tuple[float, float]] = None,
    family_separator: bool = True,
) -> Tuple[Figure, Axes]:
    """Diverging heatmap of (block x pair) Δ-mAP, centered on 0.

    When ``family_separator`` is True the rows are reordered with familiar
    blocks at the top and novel blocks below, with a horizontal line
    drawn at the family boundary.
    """
    if delta_grid.empty:
        fig, ax = plt.subplots(figsize=figsize or (6.5, 4.0))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    df = delta_grid.copy()
    if family_separator:
        fam_idx = [b for b in df.index if fa.block_family(b) == "familiar"]
        nov_idx = [b for b in df.index if fa.block_family(b) == "novel"]
        oth_idx = [b for b in df.index if b not in fam_idx and b not in nov_idx]
        df = df.loc[fam_idx + nov_idx + oth_idx]
        boundary = len(fam_idx)
    else:
        boundary = None

    if figsize is None:
        figsize = (max(6.0, 0.7 * df.shape[1] + 1.5),
                   max(3.5, 0.30 * df.shape[0] + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    if vlim is None:
        v = float(df.abs().to_numpy().max())
        vlim = v if v > 0 else 0.01
    im = _heatmap(df, ax=ax, cmap="RdBu_r",
                  vmin=-vlim, vmax=+vlim, title=title,
                  annotate=True, fmt="{:+.3f}")
    if boundary and 0 < boundary < df.shape[0]:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.0, alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Δ mAP")
    fig.tight_layout()
    return fig, ax


def plot_per_domain_bars(
    grid: pd.DataFrame,
    *,
    variants: Optional[Sequence[str]] = None,
    title: str = "Per-block end-of-training mAP",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Axes]:
    """Grouped bar chart of (block x variant) mAP."""
    if variants is not None:
        cols = [c for c in variants if c in grid.columns]
        df = grid[cols]
    else:
        df = grid
    n_blocks, n_vars = df.shape
    if figsize is None:
        figsize = (max(8.0, 0.7 * n_blocks + 2.0), 4.5)
    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    x = np.arange(n_blocks)
    bar_w = 0.9 / max(1, n_vars)
    for i, lab in enumerate(df.columns):
        ax.bar(x + i * bar_w - 0.45 + bar_w / 2,
               df[lab].values, width=bar_w, label=lab,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("mAP")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=min(len(df.columns), 4), framealpha=0.85)
    fig.tight_layout()
    return fig, ax


def plot_per_block_trajectory(
    trajectories: Mapping[str, pd.DataFrame],
    blocks: Sequence[str],
    *,
    x_col: str = "checkpoint_idx",
    n_cols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4.0, 2.5),
    title: str = "Per-block mAP trajectory",
) -> Tuple[Figure, np.ndarray]:
    """Per-block mAP-over-time, faceted with one panel per block.

    ``trajectories`` is ``{variant_name: long_df}`` from
    `federated.per_block_trajectory`.
    """
    blocks = list(blocks)
    n = len(blocks)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols,
                 figsize_per_panel[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    for idx, block in enumerate(blocks):
        ax = axes_flat[idx]
        for variant, traj in trajectories.items():
            sub = traj[traj["bucket"] == block]
            if sub.empty:
                continue
            sub = sub.sort_values(x_col) if x_col in sub.columns else sub
            ax.plot(sub[x_col], sub["mAP"],
                    color=variant_color(variant),
                    linestyle=variant_linestyle(variant),
                    label=fa.label_for(variant), linewidth=1.2, alpha=0.9)
        family = fa.block_family(block)
        face = "#fff7e6" if family == "familiar" else None
        if face is not None:
            ax.set_facecolor(face)
        ax.set_title(f"{block} ({family})", fontsize=9, loc="left")
        ax.grid(True, axis="y", alpha=0.3)
        ax.grid(False, axis="x")
        ax.set_ylabel("mAP", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(x_col.replace("_", " "), fontsize=8)
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7.5,
                   framealpha=0.9, ncol=1, bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.97, 0.97))
    return fig, axes


def plot_per_block_trajectory_delta(
    traj_delta: pd.DataFrame,
    *,
    metric: str = "cum_avg_delta",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Per-block ``cum_avg_delta`` (or ``final_delta``) for each iso-accept pair.

    ``traj_delta`` is the long-format DataFrame from
    `federated.per_block_trajectory_delta`.  Rows are
    blocks; columns are pairings.  Familiar blocks are pinned at the
    top and separated from novel blocks by a horizontal line.
    """
    if traj_delta.empty:
        fig, ax = plt.subplots(figsize=figsize or (6.5, 4.0))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    pivot = traj_delta.pivot_table(
        index="block", columns="filter_label", values=metric,
        aggfunc="mean",
    )
    fam_idx = [b for b in pivot.index if fa.block_family(b) == "familiar"]
    nov_idx = [b for b in pivot.index if fa.block_family(b) == "novel"]
    oth_idx = [b for b in pivot.index if b not in fam_idx and b not in nov_idx]
    pivot = pivot.loc[fam_idx + nov_idx + oth_idx]
    boundary = len(fam_idx)

    if figsize is None:
        figsize = (max(6.0, 0.7 * pivot.shape[1] + 1.5),
                   max(3.5, 0.30 * pivot.shape[0] + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    v = float(pivot.abs().to_numpy().max())
    vlim = v if v > 0 else 0.01
    title_txt = (title or f"Per-block {metric}\n(filter mAP $-$ iso-accept random mAP)")
    im = _heatmap(pivot, ax=ax, cmap="RdBu_r",
                  vmin=-vlim, vmax=+vlim, title=title_txt,
                  annotate=True, fmt="{:+.4f}")
    if 0 < boundary < pivot.shape[0]:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.0, alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=metric)
    fig.tight_layout()
    return fig, ax


def plot_balanced_vs_worst(
    summary: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (7.5, 4.5),
    title: str = "Balanced vs worst-block mAP per variant",
) -> Tuple[Figure, Axes]:
    """Scatter of balanced (mean) mAP vs worst-block mAP."""
    fig, ax = plt.subplots(figsize=figsize)
    if summary.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    for _, row in summary.iterrows():
        ax.scatter(row["balanced_mAP"], row["worst_block_mAP"],
                   s=70, color="#444444", marker="o", alpha=0.85,
                   edgecolor="white")
        ax.annotate(row["variant_label"],
                    (row["balanced_mAP"], row["worst_block_mAP"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Balanced (mean) per-block mAP")
    ax.set_ylabel("Worst-block mAP")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# =============================================================================
# Ablations
# =============================================================================

def plot_ablation_pair_bar(
    pair_df: pd.DataFrame,
    *,
    title: str = "Ablation pair: baseline vs ablated",
    figsize: Tuple[float, float] = (7.5, 4.0),
) -> Tuple[Figure, Axes]:
    """Side-by-side bars of baseline vs ablated smoothed mAP per pair."""
    fig, ax = plt.subplots(figsize=figsize)
    if pair_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    x = np.arange(len(pair_df))
    bar_w = 0.4
    ax.bar(x - bar_w / 2, pair_df["baseline_smoothed"], width=bar_w,
           color="#1f77b4", edgecolor="white", linewidth=0.4,
           label="baseline")
    ax.bar(x + bar_w / 2, pair_df["ablated_smoothed"], width=bar_w,
           color="#ff7f0e", edgecolor="white", linewidth=0.4,
           label="ablated")
    for xi, d in zip(x, pair_df["delta_smoothed"]):
        if pd.notna(d):
            ax.annotate(f"Δ={d:+.4f}",
                        xy=(xi, max(pair_df["baseline_smoothed"].iloc[xi],
                                    pair_df["ablated_smoothed"].iloc[xi])),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_df["pair"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Smoothed tail-5 mAP")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    return fig, ax


# =============================================================================
# Refresh dynamics (within-stream accept-rate decay diagnostic)
# =============================================================================

def plot_refresh_segment_accept(
    segments: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (8.0, 4.0),
    title: str = "Inter-refresh accept rate (single-seed diagnostic)",
) -> Tuple[Figure, Axes]:
    """Step plot of accept rate inside each inter-refresh segment per variant.

    ``segments`` is the long-format DataFrame from
    `federated.refresh_segment_table`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if segments.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    for variant, sub in segments.groupby("variant", sort=False):
        sub = sub.sort_values("segment_start")
        c = variant_color(variant)
        ax.step(sub["segment_start"], sub["accept_rate"], where="post",
                color=c, linewidth=1.4, alpha=0.9,
                label=fa.label_for(variant))
    ax.set_xlabel("decision index (global)")
    ax.set_ylabel("accept rate within segment")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.85, ncol=2)
    fig.tight_layout()
    return fig, ax
