"""
Figure generators for the streaming experiments.

All plotting calls return ``(fig, ax)`` so the notebook can save and
inline them without baking the file system path into the helper.  Use
`save_figure` (re-exported from `stream_active_fl.analysis.figures`) for
the canonical PDF + PNG export.

Public surface:

    Style helpers
        FAMILY_COLORS                 -- variant-family palette
        variant_color(variant)        -- stable color from family
        variant_linestyle(variant)    -- solid / dashed / dotted from flavor
        variant_marker(variant)       -- marker shape from flavor
        save_figure(fig, name, fmt)   -- writes PDF + PNG into figures/

    Headline figures
        plot_inventory_scatter(inv)
        plot_iso_accept_scatter(inv, pairings)
        plot_overall_mAP_trajectory(traj_df, manifest)
        plot_smoothed_leaderboard(inv)

    Per-domain figures
        plot_per_domain_heatmap(grid, *, vmin, vmax, title)
        plot_per_domain_delta_heatmap(delta_grid, ...)
        plot_per_domain_bars(grid, variants_to_show)
        plot_per_block_trajectory(per_block_traj_by_variant, blocks)
        plot_balanced_vs_worst(summary_df)

    Accept dynamics
        plot_per_block_routing(rate_grid, baseline_label)
        plot_rolling_accept_rate(rolling_by_variant, manifest, project_root)

    Ablations
        plot_ablation_pair_bar(pair_df, *, title)
        plot_static_vs_adaptive(inv)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .. import runs as ah
from .. import streaming as sa
from . import heatmap as _heatmap, save_figure  # noqa: F401  (re-export)


# =============================================================================
# Style
# =============================================================================

FAMILY_COLORS: Dict[str, str] = dict(ah.FILTER_FAMILY_COLORS)
FAMILY_COLORS["random"] = "#7f7f7f"

# Per-variant color overrides.  Distinct hues are required because the
# rolling accept-rate plot fills +/-1 std confidence bands at low alpha;
# two bands sharing a family color become impossible to tell apart.
# Variants are still loosely grouped by family hue so the legend reads
# naturally:
#
#   - window family    -> warm orange / yellow / tan
#   - reservoir family -> red / magenta / salmon
#   - static family    -> light vs dark blue
#   - random family    -> shades of grey
#   - no_filter        -> green
VARIANT_COLOR_PALETTE: Dict[str, str] = {
    # baselines
    "no_filter_cityday_curated":                          "#2ca02c",  # green
    "no_filter_cityday_temporal":                         "#2ca02c",
    "random_p17_cityday_curated":                         "#c8c8c8",
    "random_p21_cityday_curated":                         "#a0a0a0",
    "random_p21_cityday_temporal":                        "#a0a0a0",
    "random_p28_cityday_temporal":                        "#808080",
    "random_p29_cityday_curated":                         "#808080",
    "random_p31_cityday_temporal":                        "#606060",
    "random_p33_cityday_curated":                         "#606060",
    "random_p73_cityday_curated":                         "#404040",
    "random_p77_cityday_curated":                         "#202020",
    # static — purple vs blue (truly distinct hues, not just light/dark blue)
    "static_p15_cityday_curated":                         "#9467bd",  # purple
    "static_p20_cityday_curated":                         "#1f77b4",  # steel blue
    # window family — orange vs olive (warm, but different hue not just shade)
    "adaptive_window_p15_cityday_curated":                "#ffbb78",  # light tan
    "adaptive_window_p20_cityday_curated":                "#ff7f0e",  # vivid orange
    "adaptive_window_p20_twoRef_cityday_curated":         "#bcbd22",  # olive (tab10)
    "adaptive_window_p20_noBoot_cityday_curated":         "#8c564b",  # warm brown (tab10)
    "adaptive_window_p20_cityday_temporal":               "#ff7f0e",
    "adaptive_window_p20_twoRef_cityday_temporal":        "#bcbd22",
    # reservoir family — red vs pink (different hue not just shade)
    "adaptive_reservoir_p15_cityday_curated":             "#fc8d62",  # salmon
    "adaptive_reservoir_p20_cityday_curated":             "#d62728",  # vivid red
    "adaptive_reservoir_p20_twoRef_cityday_curated":      "#e377c2",  # pink (tab10)
    "adaptive_reservoir_p20_noBoot_cityday_curated":      "#8b0000",  # dark red
    "adaptive_reservoir_p20_cityday_temporal":            "#d62728",
    "adaptive_reservoir_p20_twoRef_cityday_temporal":     "#e377c2",
}


def variant_color(variant: str, *, project_root: Optional[Path] = None) -> str:
    """Return a stable matplotlib color for a variant.

    Each *featured* variant in `VARIANT_COLOR_PALETTE` has a hand-picked
    distinct hue; everything else falls back to the family-level color
    so unfamiliar variants still render plausibly.
    """
    if variant in VARIANT_COLOR_PALETTE:
        return VARIANT_COLOR_PALETTE[variant]
    fam = sa.family_for_variant(variant, project_root=project_root)
    return FAMILY_COLORS.get(fam, "#444444")


def variant_linestyle(variant: str) -> str:
    """Return a stable matplotlib linestyle from the variant flavor.

    The color is set by family; the linestyle distinguishes flavors
    *within* a family.  Convention:

    - twoRef       -> ``--`` (dashed)
    - noBoot       -> ``:``  (dotted)
    - p15 budget   -> ``--`` (dashed) when no other flavor applies, so the
                       "headline" p20 variant remains solid in any plot
                       that mixes p15 and p20 within a family
    - otherwise    -> ``-``  (solid)
    """
    if "twoRef" in variant:
        return "--"
    if "noBoot" in variant:
        return ":"
    if "_p15_" in variant:
        return "--"
    return "-"


def variant_marker(variant: str) -> str:
    """Return a stable marker shape from variant flavor (twoRef / noBoot / vanilla)."""
    if "twoRef" in variant:
        return "s"
    if "noBoot" in variant:
        return "X"
    if variant.startswith("static_"):
        return "D"
    if variant.startswith("random_"):
        return "."
    if variant.startswith("no_filter"):
        return "*"
    return "o"


# =============================================================================
# Headline figures
# =============================================================================

def plot_inventory_scatter(
    inv: pd.DataFrame,
    *,
    manifest: str = "curated",
    figsize: Tuple[float, float] = (7.5, 5.0),
) -> Tuple[Figure, Axes]:
    """Scatter of accept_rate (x) vs smoothed_mAP (y) with random-baseline curve.

    Random baselines for the same manifest are joined into a reference
    curve so the eye can read "where would a random filter at this accept
    rate land".  Filters above the curve outperform iso-accept random.
    """
    sub = inv[inv["manifest"] == manifest].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)

    # Random reference curve
    rand = sub[sub["family"] == "random"].sort_values("accept_rate")
    if not rand.empty:
        ax.plot(rand["accept_rate"], rand["smoothed_mAP"],
                color=FAMILY_COLORS.get("random", "#7f7f7f"),
                linestyle="--", linewidth=1.2, alpha=0.8,
                label="random envelope", zorder=1)

    # All variants as scatter, colored by family with seed-std error bars
    for _, row in sub.iterrows():
        ax.errorbar(
            row["accept_rate"], row["smoothed_mAP"],
            yerr=row["smoothed_std"] if pd.notna(row["smoothed_std"]) else 0,
            fmt=variant_marker(row["variant"]),
            color=variant_color(row["variant"]),
            markersize=8, markeredgecolor="white", markeredgewidth=0.6,
            capsize=2, elinewidth=0.6,
            alpha=0.95, zorder=3,
        )
        ax.annotate(row["label"], (row["accept_rate"], row["smoothed_mAP"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color="#333333")

    ax.set_xlabel("Effective accept rate")
    ax.set_ylabel("Smoothed tail-5 mAP")
    ax.set_title(f"Iso-accept leaderboard — {manifest} stream",
                 fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)

    # Family legend
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
    figsize: Tuple[float, float] = (7.5, 5.0),
) -> Tuple[Figure, Axes]:
    """Plot Δsmoothed mAP vs accept_gap for each filter <-> random pair.

    A point in the upper-half-plane is a filter that beats iso-accept
    random (positive Δ); points near x=0 are within the iso-accept band.
    """
    sub = iso[iso["manifest"] == manifest].copy()
    fig, ax = plt.subplots(figsize=figsize)
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    for _, row in sub.iterrows():
        ax.errorbar(
            row["filter_accept"],
            row["delta_smoothed"],
            yerr=np.sqrt((row["filter_smoothed_std"] or 0) ** 2 +
                         (row["random_smoothed_std"] or 0) ** 2),
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
    ax.set_title(f"Iso-accept gain — {manifest} stream",
                 fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_overall_mAP_trajectory(
    traj: pd.DataFrame,
    *,
    x_col: str = "items_processed",
    figsize: Tuple[float, float] = (8.0, 4.5),
    block_transitions: Optional[Sequence[Tuple[int, str]]] = None,
    title: str = "Streaming mAP",
) -> Tuple[Figure, Axes]:
    """Plot overall mAP vs ``x_col`` for multiple variants (already aggregated).

    ``traj`` must come from `streaming.mAP_trajectory`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if traj.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    if block_transitions:
        for x, _lab in block_transitions[1:]:  # skip the first edge (start of stream)
            ax.axvline(x, color="#444444", alpha=0.35, lw=0.6, ls="--", zorder=0)

    for v, grp in traj.groupby("variant", sort=False):
        grp = grp.sort_values(x_col)
        c = variant_color(v)
        ax.plot(grp[x_col], grp["mAP"], color=c, label=sa.label_for(v),
                linewidth=1.4, alpha=0.9)
        if "mAP_std" in grp.columns:
            ax.fill_between(grp[x_col],
                            grp["mAP"] - grp["mAP_std"].fillna(0),
                            grp["mAP"] + grp["mAP_std"].fillna(0),
                            color=c, alpha=0.12, linewidth=0)

    ax.set_xlabel("items processed" if x_col == "items_processed" else x_col)
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
    sort_by: str = "smoothed_mAP",
    figsize: Tuple[float, float] = (8.0, 5.0),
) -> Tuple[Figure, Axes]:
    """Horizontal bar chart of smoothed-tail mAP, sorted descending."""
    sub = inv[inv["manifest"] == manifest].copy()
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
    ax.set_title(f"{sort_by.replace('_', ' ')} — {manifest}",
                 fontsize=11, loc="left")
    ax.grid(True, axis="x", alpha=0.3)
    # Annotate accept rate on the right of each bar
    for bar, ar in zip(bars, sub["accept_rate"]):
        ax.text(bar.get_width() + 0.0008, bar.get_y() + bar.get_height() / 2,
                f"acc={ar:.2f}", va="center", fontsize=7, color="#333333")
    fig.tight_layout()
    return fig, ax


# =============================================================================
# Per-domain figures
# =============================================================================


def plot_per_domain_heatmap(
    grid: pd.DataFrame,
    *,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Per-domain end-of-stream mAP",
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
    title: str = r"Per-domain $\Delta$mAP (filter $-$ iso-accept random)",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Axes]:
    """Diverging heatmap of (block x pair) Δmap deltas, centered on 0."""
    if figsize is None:
        figsize = (max(6.0, 0.7 * delta_grid.shape[1] + 1.5),
                   max(3.5, 0.30 * delta_grid.shape[0] + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    if delta_grid.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    if vlim is None:
        v = float(delta_grid.abs().to_numpy().max())
        vlim = v if v > 0 else 0.01
    im = _heatmap(delta_grid, ax=ax, cmap="RdBu_r",
                  vmin=-vlim, vmax=+vlim, title=title,
                  annotate=True, fmt="{:+.3f}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Δ mAP")
    fig.tight_layout()
    return fig, ax


def plot_per_domain_bars(
    grid: pd.DataFrame,
    *,
    variants: Optional[Sequence[str]] = None,
    title: str = "Per-domain end-of-stream mAP",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Axes]:
    """Grouped bar chart of (block x variant) mAP, optionally sub-selecting variants."""
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
    x_col: str = "items_processed",
    n_cols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4.0, 2.5),
    title: str = "Per-block mAP trajectory",
) -> Tuple[Figure, np.ndarray]:
    """Per-block mAP-over-time, faceted with one panel per block.

    ``trajectories`` is ``{variant_name: long_df}`` from
    `streaming.per_domain_trajectory`.  Each ``long_df`` has
    columns ``checkpoint_idx, items_processed, optimizer_steps, bucket, mAP``.
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
            ax.plot(sub[x_col], sub["mAP"], color=variant_color(variant),
                    label=sa.label_for(variant), linewidth=1.2, alpha=0.9)
        ax.set_title(block, fontsize=9, loc="left")
        ax.grid(True, axis="y", alpha=0.3)
        ax.grid(False, axis="x")
        ax.set_ylabel("mAP", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(x_col.replace("_", " "), fontsize=8)
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")
    # Single shared legend on the right
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7.5,
                   framealpha=0.9, ncol=1, bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.97, 0.97))
    return fig, axes


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
        v = next((x for x in sa.FEATURED_VARIANTS if sa.label_for(x) == row["variant_label"]),
                 row["variant_label"])
        ax.scatter(row["balanced_mAP"], row["worst_block_mAP"],
                   color=variant_color(v),
                   marker=variant_marker(v),
                   s=64, edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(row["variant_label"],
                    (row["balanced_mAP"], row["worst_block_mAP"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    # Diagonal y=x for reference (worst <= balanced always)
    lo = min(summary["worst_block_mAP"].min(), summary["balanced_mAP"].min())
    hi = max(summary["worst_block_mAP"].max(), summary["balanced_mAP"].max())
    pad = 0.01
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="#888888", linestyle=":", linewidth=0.8, alpha=0.7,
            label="y = x")
    ax.set_xlabel("Balanced mAP (mean over blocks)")
    ax.set_ylabel("Worst-block mAP")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# =============================================================================
# Accept dynamics
# =============================================================================

def plot_per_block_routing(
    rate_grid: pd.DataFrame,
    *,
    baseline_label: Optional[str] = "rand_p33",
    title: str = "Per-block accept rate",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Axes]:
    """Heatmap of per-block accept rate, optionally normalised by a random row."""
    if figsize is None:
        figsize = (max(6.5, 0.45 * rate_grid.shape[1] + 1.5),
                   max(3.5, 0.30 * rate_grid.shape[0] + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    if rate_grid.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    im = _heatmap(rate_grid, ax=ax, cmap="magma",
                  vmin=0.0, vmax=min(1.0, max(0.6, rate_grid.to_numpy().max() + 0.05)),
                  title=title, annotate=True, fmt="{:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="accept rate")
    fig.tight_layout()
    return fig, ax


def _draw_block_lines(
    ax: Axes,
    boundaries: Sequence[int],
    *,
    color: str = "#444444",
    alpha: float = 0.35,
    linewidth: float = 0.6,
    linestyle: str = "--",
) -> None:
    """Draw dashed vertical lines at block transitions only.

    ``boundaries[1:-1]`` are the inter-block transitions (excluding the
    stream start and end).  Use this everywhere the x-axis is "stream
    index" so the only x-axis annotation marks domain shifts.
    """
    for x in list(boundaries)[1:-1]:
        ax.axvline(x, color=color, alpha=alpha, lw=linewidth, ls=linestyle, zorder=0)


def _annotate_block_numbers(
    ax: Axes,
    midpoints: Sequence[Tuple[int, str]],
) -> None:
    """Place 1-based block numbers along the top of an axes (xaxis transform)."""
    for i, (xm, _label) in enumerate(midpoints, start=1):
        ax.text(xm, 1.01, str(i), ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#444444", alpha=0.9,
                transform=ax.get_xaxis_transform())


def plot_rolling_accept_rate(
    accept_by_variant: Mapping[str, pd.DataFrame],
    *,
    boundaries: Sequence[int] = (),
    midpoints: Sequence[Tuple[int, str]] = (),
    composition: Optional[Mapping[str, pd.DataFrame]] = None,
    composition_palettes: Optional[Mapping[str, Mapping[str, str]]] = None,
    composition_titles: Optional[Mapping[str, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Accept rate per stream window",
    window: int = 1000,
) -> Tuple[Figure, np.ndarray]:
    """Multi-panel accept-rate-with-stream-composition plot.

    Top panel: per-variant accept rate per window, mean +/- std band over
    seeds.  Optional lower panels: stacked-area composition bars (e.g.
    time-of-day, road-condition) describing what *kind* of frames the
    stream is presenting in each window.

    Args:
        accept_by_variant: `{variant: df}` from
            `streaming.windowed_accept_rate_aggregated`, with columns
            `items_start`, `accept_rate_mean`, `accept_rate_std`.
        boundaries: Stream-block boundaries from
            `streaming.block_boundaries_and_midpoints`.
        midpoints: Stream-block midpoints from the same helper.
        composition: `{field_name: wide_df}` where each wide_df has a
            row per `items_start` and one column per category, holding
            fractions in [0, 1].  Each entry becomes one stacked-area
            panel.
    """
    has_comp = composition and any(not df.empty for df in composition.values())
    n_comp = sum(1 for df in (composition or {}).values() if not df.empty)
    n_panels = 1 + n_comp

    if figsize is None:
        figsize = (12.5, 3.0 + 1.6 * n_comp)

    height_ratios = [2.4] + [1.0] * n_comp
    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )
    axes = axes[:, 0]

    # Top panel: accept rate
    ax = axes[0]
    has_data = False
    for variant, df in accept_by_variant.items():
        if df is None or df.empty:
            continue
        has_data = True
        xs = df["items_start"].to_numpy()
        means = df["accept_rate_mean"].to_numpy()
        stds = df.get("accept_rate_std",
                      pd.Series(np.zeros_like(means))).fillna(0.0).to_numpy()
        ax.plot(xs, means, color=variant_color(variant),
                lw=1.5, alpha=0.95, ls=variant_linestyle(variant),
                label=sa.label_for(variant), zorder=3)
        ax.fill_between(xs, means - stds, means + stds,
                        color=variant_color(variant), alpha=0.13,
                        linewidth=0, zorder=2)
    if not has_data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
    if boundaries:
        _draw_block_lines(ax, boundaries)
    if midpoints:
        _annotate_block_numbers(ax, midpoints)
    ax.set_ylabel("Accept rate")
    ax.set_ylim(bottom=0.0)
    ax.set_title(title, fontsize=10, loc="left", pad=14)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")
    if has_data:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92,
                  ncol=min(3, max(1, len(accept_by_variant))))

    # Composition panels
    if has_comp:
        for ax, (field, frac_df) in zip(axes[1:], composition.items()):
            if frac_df.empty:
                ax.set_visible(False)
                continue
            xs = frac_df.index.to_numpy()
            palette = (composition_palettes or {}).get(field, {})
            base = np.zeros_like(xs, dtype=float)
            for col in frac_df.columns:
                color = palette.get(col, None)
                vals = frac_df[col].to_numpy()
                if color is not None:
                    ax.fill_between(xs, base, base + vals,
                                    color=color, alpha=0.75, label=col,
                                    linewidth=0)
                else:
                    ax.fill_between(xs, base, base + vals,
                                    alpha=0.65, label=col, linewidth=0)
                base = base + vals
            if boundaries:
                _draw_block_lines(ax, boundaries)
            ax.set_ylabel("Fraction")
            ax.set_ylim(0, 1)
            ttl = (composition_titles or {}).get(field, f"{field} composition")
            ax.set_title(ttl, fontsize=10, loc="left")
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92,
                      ncol=min(4, max(1, len(frac_df.columns))))
            ax.grid(False)

    axes[-1].set_xlabel("Frame index (post-bootstrap)")
    if boundaries:
        axes[-1].set_xlim(boundaries[0], boundaries[-1])

    fig.tight_layout()
    return fig, axes


# =============================================================================
# Ablations
# =============================================================================

def plot_ablation_pair_bar(
    pair_df: pd.DataFrame,
    *,
    title: str = "Ablation pairs",
    figsize: Tuple[float, float] = (7.0, 3.5),
) -> Tuple[Figure, Axes]:
    """Side-by-side bars: baseline_smoothed vs ablated_smoothed.

    Annotates each pair with the delta (ablated - baseline).
    """
    fig, ax = plt.subplots(figsize=figsize)
    if pair_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    n = len(pair_df)
    x = np.arange(n)
    w = 0.4
    base_color = "#1f77b4"
    abl_color = "#ff7f0e"
    ax.bar(x - w / 2, pair_df["baseline_smoothed"], width=w,
           color=base_color, label="baseline", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, pair_df["ablated_smoothed"], width=w,
           color=abl_color, label="ablated",  edgecolor="white", linewidth=0.5)
    for i, row in pair_df.reset_index(drop=True).iterrows():
        ax.text(i, max(row["baseline_smoothed"], row["ablated_smoothed"]) + 0.0015,
                f"Δ={row['delta_smoothed']:+.4f}",
                ha="center", fontsize=7.5,
                color=("#1a7f1a" if row["delta_smoothed"] > 0 else "#7f1a1a"))
    ax.set_xticks(x)
    ax.set_xticklabels(pair_df["pair"], rotation=10, ha="right", fontsize=8)
    ax.set_ylabel("smoothed tail-5 mAP")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_per_class_heatmap(
    grid: pd.DataFrame,
    *,
    title: str = "Per-class end-of-stream AP",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Axes]:
    """Heatmap of the (class x variant) per-class AP grid."""
    return plot_per_domain_heatmap(grid, title=title, figsize=figsize)


def plot_per_class_trajectory(
    trajectories: Mapping[str, pd.DataFrame],
    classes: Sequence[str],
    *,
    x_col: str = "items_processed",
    figsize_per_panel: Tuple[float, float] = (4.2, 2.5),
    title: str = "Per-class AP trajectory",
) -> Tuple[Figure, np.ndarray]:
    """Per-class AP-over-stream, faceted with one panel per class."""
    classes = list(classes)
    n = len(classes)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    for idx, cls in enumerate(classes):
        ax = axes_flat[idx]
        for variant, traj in trajectories.items():
            sub = traj[traj["class"] == cls]
            if sub.empty:
                continue
            sub = sub.sort_values(x_col) if x_col in sub.columns else sub
            ax.plot(sub[x_col], sub["AP"], color=variant_color(variant),
                    label=sa.label_for(variant), linewidth=1.3, alpha=0.9)
        ax.set_title(cls, fontsize=10, loc="left")
        ax.grid(True, axis="y", alpha=0.3)
        ax.grid(False, axis="x")
        ax.set_ylabel("AP", fontsize=8)
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


def plot_forgetting_heatmap(
    forget: pd.DataFrame,
    *,
    metric: str = "delta",
    title: str = r"Forgetting: $\Delta$ AP (last $-$ first quartile)",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Axes]:
    """Pivot the forgetting table into a (variant x class) heatmap.

    ``metric`` is one of ``delta``, ``early`` or ``late``.  ``delta`` uses
    a diverging colormap centred on zero; ``early``/``late`` use viridis.
    """
    if forget.empty or metric not in forget.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    grid = forget.pivot(index="label", columns="class", values=metric)
    if figsize is None:
        figsize = (max(5.0, 0.7 * grid.shape[1] + 1.5),
                   max(3.5, 0.30 * grid.shape[0] + 1.2))
    fig, ax = plt.subplots(figsize=figsize)
    if metric == "delta":
        v = float(grid.abs().to_numpy().max())
        v = v if v > 0 else 0.01
        im = _heatmap(grid, ax=ax, cmap="RdBu_r",
                      vmin=-v, vmax=+v, title=title,
                      annotate=True, fmt="{:+.3f}")
    else:
        im = _heatmap(grid, ax=ax, cmap="viridis", title=title,
                      annotate=True, fmt="{:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=metric)
    fig.tight_layout()
    return fig, ax


def plot_refresh_segment_decay(
    seg: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (8.5, 4.0),
    title: str = "Per-refresh accept rate (within-segment decay)",
) -> Tuple[Figure, Axes]:
    """Line plot of accept rate vs refresh index for each variant."""
    fig, ax = plt.subplots(figsize=figsize)
    if seg.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    for variant, grp in seg.groupby("variant", sort=False):
        grp = grp.sort_values("refresh_idx")
        ax.plot(grp["refresh_idx"], grp["accept_rate"],
                marker="o", markersize=4, linewidth=1.2,
                color=variant_color(variant),
                label=sa.label_for(variant), alpha=0.9)
    ax.set_xlabel("refresh index (-1 = pre-first-refresh)")
    ax.set_ylabel("segment accept rate")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7.5, framealpha=0.85)
    fig.tight_layout()
    return fig, ax


def plot_steps_to_target(
    steps: pd.DataFrame,
    *,
    x_col: str = "optimizer_steps",
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Steps to reach a target mAP",
) -> Tuple[Figure, Axes]:
    """Grouped bar chart: target_mAP on x, optimizer_steps on y, bars per variant."""
    if steps.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    pivot = steps.pivot(index="target_mAP", columns="label", values=x_col)
    if figsize is None:
        figsize = (max(7.0, 0.6 * pivot.shape[0] + 1.0 + 0.3 * pivot.shape[1]),
                   4.5)
    fig, ax = plt.subplots(figsize=figsize)
    n_t, n_v = pivot.shape
    bar_w = 0.85 / max(1, n_v)
    x = np.arange(n_t)
    for i, lab in enumerate(pivot.columns):
        v = next((var for var in sa.FEATURED_VARIANTS if sa.label_for(var) == lab), lab)
        ax.bar(x + i * bar_w - 0.4 + bar_w / 2,
               pivot[lab].values, width=bar_w, label=lab,
               color=variant_color(v), edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.3f}" for t in pivot.index], fontsize=8)
    ax.set_xlabel("target mAP")
    ax.set_ylabel(x_col.replace("_", " "))
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=min(len(pivot.columns), 4),
              framealpha=0.85, loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_static_vs_adaptive(
    inv: pd.DataFrame,
    *,
    manifest: str = "curated",
    figsize: Tuple[float, float] = (6.5, 4.0),
) -> Tuple[Figure, Axes]:
    """Bar chart positioning static_p15/p20 against adaptive p20 variants and randoms."""
    keep_labels = [
        "no_filter", "rand_p33", "rand_p77",
        "static_p15", "static_p20",
        "Win_p20", "Win_p20_twoRef", "Res_p20", "Res_p20_twoRef",
    ]
    sub = inv[(inv["manifest"] == manifest) &
              inv["label"].isin(keep_labels)].copy()
    sub["order"] = sub["label"].map({lab: i for i, lab in enumerate(keep_labels)})
    sub = sub.sort_values("order")
    fig, ax = plt.subplots(figsize=figsize)
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    colors = [variant_color(v) for v in sub["variant"]]
    bars = ax.bar(sub["label"], sub["smoothed_mAP"], color=colors,
                  yerr=sub["smoothed_std"], capsize=2,
                  edgecolor="white", linewidth=0.5)
    for bar, ar in zip(bars, sub["accept_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{ar:.2f}", ha="center", fontsize=7, color="#333")
    ax.set_ylabel("smoothed tail-5 mAP")
    ax.set_xlabel("(numbers above bars: empirical accept rate)")
    ax.set_title(f"Static vs adaptive at p20 — {manifest}",
                 fontsize=11, loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    return fig, ax
