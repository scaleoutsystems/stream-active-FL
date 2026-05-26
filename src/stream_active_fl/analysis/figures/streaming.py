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

# Single-hue-per-family palette.  Family is the only thing color
# encodes; flavor (single-ref vs two-ref vs no-anchor vs heavy-local)
# is encoded by linestyle + marker.  This keeps figures readable in
# greyscale print and removes confusable hues (e.g. olive vs green).
FAMILY_COLORS: Dict[str, str] = {
    "none":       "#2ca02c",  # no-filter ceiling -- green
    "no_filter":  "#2ca02c",  # alias for the no-filter family
    "random":     "#7f7f7f",  # neutral grey
    "static":     "#1f77b4",  # blue
    "window":     "#ff7f0e",  # orange
    "reservoir":  "#d62728",  # red
}

# Random baselines need to be distinguishable when several appear in
# the same scatter; vary lightness, not hue.  Lower accept fractions
# get lighter shades.  Falls back to FAMILY_COLORS["random"] for
# unmapped randoms.
_RANDOM_GREYSCALE: Dict[str, str] = {
    "random_p17_cityday_curated":   "#c4c4c4",
    "random_p21_cityday_curated":   "#aaaaaa",
    "random_p23_cityday_curated":   "#9a9a9a",
    "random_p26_cityday_curated":   "#878787",
    "random_p27_cityday_curated":   "#787878",
    "random_p29_cityday_curated":   "#686868",
    "random_p33_cityday_curated":   "#555555",
    "random_p73_cityday_curated":   "#383838",
    "random_p77_cityday_curated":   "#1f1f1f",
    "random_p21_cityday_temporal":  "#aaaaaa",
    "random_p28_cityday_temporal":  "#787878",
    "random_p31_cityday_temporal":  "#555555",
}


def variant_color(variant: str, *, project_root: Optional[Path] = None) -> str:
    """Return the family-level color for a variant.

    All members of a family share the same hue; flavor is conveyed by
    linestyle + marker.  The exception is the random family, where
    different accept fractions are drawn in different shades of grey
    so iso-accept comparisons remain readable on a single axis.
    """
    if variant in _RANDOM_GREYSCALE:
        return _RANDOM_GREYSCALE[variant]
    fam = sa.family_for_variant(variant, project_root=project_root)
    return FAMILY_COLORS.get(fam, "#444444")


def variant_linestyle(variant: str) -> str:
    """Return a stable matplotlib linestyle from the variant flavor.

    Encoding (family-agnostic):

    - solid ``-``  : headline / single-reference variants and the
                     reference baselines (no-filter, random, static
                     ``\\tau_{20}``).
    - dashed ``--``: the two-reference Mahalanobis variant.
    - dotted ``:`` : the bootstrap-only ablation (noBoot).
    - dashed ``--``: the lower-percentile ``static_p15`` variant when
                     paired with ``static_p20``.
    """
    if "twoRef" in variant:
        return "--"
    if "noBoot" in variant:
        return ":"
    if variant == "static_p15_cityday_curated":
        return "--"
    return "-"


def variant_marker(variant: str) -> str:
    """Return a stable marker shape encoding the variant flavor.

    - ``o``: vanilla / single-ref (default)
    - ``s``: two-reference Mahalanobis
    - ``X``: bootstrap-only ablation (noBoot)
    - ``D``: static threshold filter
    - ``*``: no-filter ceiling
    - ``.``: random baseline
    """
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
    headline_variants: Optional[Sequence[str]] = None,
    annotate_filters_only: bool = True,
    figsize: Tuple[float, float] = (7.0, 4.6),
) -> Tuple[Figure, Axes]:
    """Scatter of accept_rate (x) vs smoothed_mAP (y) with random-baseline curve.

    The full set of randoms is always used to draw the iso-accept
    envelope (the dotted grey curve); a filter sitting above the curve
    beats iso-accept random.  ``headline_variants`` restricts the
    plotted *points* to the requested subset (defaults to all variants
    on the manifest, which is what the appendix figure wants).  When
    ``annotate_filters_only`` is True, random and no-filter markers
    carry no per-point label -- the dotted random curve and the
    no-filter star are self-explanatory and clutter slows the eye.
    """
    sub = inv[inv["manifest"] == manifest].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)

    # Random reference curve always drawn from *all* randoms in `inv`.
    # The dotted polyline is the visual envelope; small grey markers
    # with vertical error bars show that each segment of the envelope
    # is a real (mean +/- seed-std) measurement, not an interpolation.
    rand = sub[sub["family"] == "random"].sort_values("accept_rate")
    if not rand.empty:
        rand_color = FAMILY_COLORS.get("random", "#7f7f7f")
        ax.plot(rand["accept_rate"], rand["smoothed_mAP"],
                color=rand_color,
                linestyle=":", linewidth=1.2, alpha=0.8,
                label="iso-accept random envelope", zorder=1)
        # Per-random markers + error bars so the envelope visibly
        # decomposes into measured data points.  Smaller than the
        # filter markers and slightly transparent so the filter
        # markers stay the visual focus.
        for _, row in rand.iterrows():
            ax.errorbar(
                row["accept_rate"], row["smoothed_mAP"],
                yerr=row["smoothed_std"] if pd.notna(row["smoothed_std"]) else 0,
                fmt=".", color=rand_color,
                markersize=5,
                capsize=2, elinewidth=0.6, alpha=0.7, zorder=2,
            )

    # Restrict points to the headline subset if requested.  Random
    # points are already drawn above as part of the envelope, so we
    # exclude them from the headline scatter to avoid double-marking.
    if headline_variants is not None:
        sub_pts = sub[sub["variant"].isin(list(headline_variants))]
    else:
        sub_pts = sub
    sub_pts = sub_pts[sub_pts["family"] != "random"]

    for _, row in sub_pts.iterrows():
        ax.errorbar(
            row["accept_rate"], row["smoothed_mAP"],
            yerr=row["smoothed_std"] if pd.notna(row["smoothed_std"]) else 0,
            fmt=variant_marker(row["variant"]),
            color=variant_color(row["variant"]),
            markersize=8, markeredgecolor="white", markeredgewidth=0.6,
            capsize=2, elinewidth=0.6,
            alpha=0.95, zorder=3,
        )
        skip = (annotate_filters_only
                and row["family"] in {"random", "none"})
        if not skip:
            ax.annotate(row["label"], (row["accept_rate"], row["smoothed_mAP"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=8, color="#333333")

    ax.set_xlabel("Effective accept rate", fontsize=11)
    ax.set_ylabel("Smoothed tail-5 mAP", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.3)

    # Family legend (one swatch per family that appears in the plot).
    fams_in_plot = [f for f in ["none", "static", "window", "reservoir", "random"]
                    if f in set(sub_pts["family"]) | set(rand["family"])]
    handles = [mpatches.Patch(color=FAMILY_COLORS.get(f, "#444"),
                              label={"none": "no-filter"}.get(f, f))
               for f in fams_in_plot]
    ax.legend(handles=handles, loc="lower right", fontsize=10, framealpha=0.9)

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
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_overall_mAP_trajectory(
    traj: pd.DataFrame,
    *,
    x_col: str = "items_processed",
    figsize: Tuple[float, float] = (8.0, 4.5),
    block_transitions: Optional[Sequence[Tuple[int, str]]] = None,
    title: str = "",
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
                linestyle=variant_linestyle(v),
                linewidth=1.4, alpha=0.9)
        if "mAP_std" in grp.columns:
            ax.fill_between(grp[x_col],
                            grp["mAP"] - grp["mAP_std"].fillna(0),
                            grp["mAP"] + grp["mAP_std"].fillna(0),
                            color=c, alpha=0.12, linewidth=0)

    ax.set_xlabel(
        "Frame index (post-bootstrap)"
        if x_col == "items_processed" else x_col,
        fontsize=11,
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.set_ylabel("mAP", fontsize=11)
    if title:
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
    title: str = "",
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
    title: str = "",
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
    if title:
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
    n_cols: int = 2,
    figsize_per_panel: Tuple[float, float] = (3.6, 2.4),
    smoothing_window: int = 1,
    title: str = "",
    active_intervals: Optional[Mapping[str, Sequence[Tuple[float, float]]]] = None,
    active_label: str = "block active in stream",
) -> Tuple[Figure, np.ndarray]:
    """Per-block mAP-over-time, faceted with one panel per block.

    ``trajectories`` is ``{variant: long_df}`` from
    :func:`streaming.per_domain_trajectory`; each ``long_df`` has
    columns ``checkpoint_idx, items_processed, optimizer_steps, bucket,
    mAP, mAP_std, n``.

    A ``smoothing_window > 1`` applies a centred rolling mean to
    ``mAP`` (and matching square-root smoothing to ``mAP_std``) for
    display only; the underlying CSV exports stay raw.

    Args:
        active_intervals: optional ``{block_name: [(x_lo, x_hi), ...]}``
            map of stream-coordinate intervals to highlight on each
            panel as a faint vertical span.  Used to mark *when in the
            stream* the panel's bucket is being presented to the model
            (e.g. ``city_night`` is on stream from items 37,674-43,011)
            so the reader can see the "before / during / after"
            adaptation behavior at a glance.

    Cross-seed std bands (``mAP +/- mAP_std``) are drawn for every
    variant at ``alpha=0.13`` (matching the federated trajectory
    figures).  Smoothing rolling-means the std but does **not** divide
    it by ``sqrt(window)``: ``mAP_std`` is the cross-seed std at each
    checkpoint, not within-checkpoint sampling noise that smoothing
    would reduce, so propagating the std unscaled is the honest
    visualization.
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
        # Active-block shading underneath the lines (zorder=0).  We
        # draw it first so even thin lines remain readable on top.
        intervals = (active_intervals or {}).get(block, ())
        for j, (lo, hi) in enumerate(intervals):
            ax.axvspan(lo, hi, color="#a8a8a8", alpha=0.30,
                       linewidth=0, zorder=0,
                       label=active_label if (idx == 0 and j == 0) else None)
        for variant, traj in trajectories.items():
            sub = traj[traj["bucket"] == block]
            if sub.empty:
                continue
            sub = sub.sort_values(x_col) if x_col in sub.columns else sub
            xs = sub[x_col].to_numpy()
            ys = sub["mAP"].to_numpy()
            ys_std = (sub["mAP_std"].fillna(0.0).to_numpy()
                      if "mAP_std" in sub.columns
                      else np.zeros_like(ys))
            if smoothing_window > 1:
                w = int(smoothing_window)
                ys = pd.Series(ys).rolling(w, center=True, min_periods=1).mean().to_numpy()
                ys_std = pd.Series(ys_std).rolling(w, center=True, min_periods=1).mean().to_numpy()
            ls = variant_linestyle(variant)
            c = variant_color(variant)
            if ys_std.any():
                ax.fill_between(xs, ys - ys_std, ys + ys_std,
                                color=c, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(xs, ys, color=c, linestyle=ls,
                    label=sa.label_for(variant), linewidth=1.3, alpha=0.95,
                    zorder=3)
        ax.set_title(block, fontsize=11, loc="left")
        ax.grid(True, axis="y", alpha=0.3)
        ax.grid(False, axis="x")
        ax.set_ylabel("mAP", fontsize=10)
        ax.tick_params(labelsize=9)
        if idx >= (n_rows - 1) * n_cols:
            label = ("Frame index (post-bootstrap)"
                     if x_col == "items_processed"
                     else x_col.replace("_", " "))
            ax.set_xlabel(label, fontsize=10)
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", fontsize=10,
                   framealpha=0.9, ncol=min(len(handles), 4),
                   bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1.0, 0.98))
    return fig, axes


def plot_balanced_vs_worst(
    summary: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (7.5, 4.5),
    title: str = "",
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
    if title:
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
    title: str = "",
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


def plot_per_block_routing_lines(
    rate_grid: pd.DataFrame,
    *,
    filter_variants: Sequence[str],
    random_refs: Optional[Mapping[str, float]] = None,
    static_variant: Optional[str] = None,
    title: str = "",
    figsize: Tuple[float, float] = (8.0, 5.2),
    ymax: float = 0.6,
    std_grid: Optional[pd.DataFrame] = None,
) -> Tuple[Figure, Axes]:
    """Per-block accept rate as a line plot (alternative to ``plot_per_block_routing``).

    ``rate_grid`` is the wide-format DataFrame produced by
    :func:`streaming.per_block_routing`: one row per block, one column
    per variant, values are accept rates.  This view is preferred over
    a heatmap when the central narrative is "filter X dips at block 5,
    peaks at block 9": each filter is drawn as its own polyline along
    the block axis, so adjacent-block changes are read directly.

    Args:
        filter_variants: variants to draw as full polylines (typically
            the four headline filter variants).
        random_refs: ``{label: value}`` map of horizontal reference
            lines (e.g. ``{"Random p21": 0.21, "Random p33": 0.33}``).
            Drawn as faint grey horizontal lines because per-block
            random accept rates are uniform by construction.
        static_variant: if given, draw the static-filter trace on a
            twinned right-hand axis (typically 0.6-1.0) so the filter
            polylines below stay zoomed in.
        std_grid: Optional wide-format DataFrame matching ``rate_grid``
            (same index / columns) carrying the per-(block, variant)
            cross-seed std.  When provided, a faint shaded band
            (``alpha=0.10``) is drawn around each filter polyline
            (and the static trace on the twinned axis).  Mirrors the
            federated per-block accept figure for visual parity.  Get
            it via ``per_block_routing(..., return_std=True)``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if rate_grid.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    blocks = list(rate_grid.index)
    x = np.arange(len(blocks))

    # Random reference horizontal lines (random accept rate is uniform
    # across blocks by construction; one number per random suffices).
    # Render with progressively darker greys so multiple randoms are
    # distinguishable in the legend.
    random_handles: List = []
    if random_refs:
        ref_items = list(random_refs.items())
        for i, (ref_label, ref_value) in enumerate(ref_items):
            shade = ["#a8a8a8", "#7a7a7a", "#525252"][min(i, 2)]
            line = ax.axhline(ref_value, color=shade, linestyle=":",
                              linewidth=1.1, alpha=0.85, label=ref_label)
            random_handles.append(line)

    # `rate_grid` columns are print-friendly variant labels (see
    # ``streaming.per_block_routing``); look the variant up by label.
    handles, labels = [], []
    for v in filter_variants:
        col = sa.label_for(v)
        if col not in rate_grid.columns:
            continue
        ys = rate_grid[col].values
        c = variant_color(v)
        if (std_grid is not None and not std_grid.empty
                and col in std_grid.columns):
            std = std_grid[col].fillna(0.0).values
            if std.any():
                ax.fill_between(x, ys - std, ys + std,
                                color=c, alpha=0.10, linewidth=0, zorder=2)
        line, = ax.plot(
            x, ys,
            color=c,
            linestyle=variant_linestyle(v),
            marker=variant_marker(v),
            markersize=5, markeredgecolor="white", markeredgewidth=0.4,
            linewidth=1.4, alpha=0.95, zorder=3,
        )
        handles.append(line)
        labels.append(sa.label_for(v))

    # Static on twinned axis (0.5-1.0) when requested.
    static_col = sa.label_for(static_variant) if static_variant else None
    if static_col is not None and static_col in rate_grid.columns:
        ax2 = ax.twinx()
        ys = rate_grid[static_col].values
        c = variant_color(static_variant)
        if (std_grid is not None and not std_grid.empty
                and static_col in std_grid.columns):
            std = std_grid[static_col].fillna(0.0).values
            if std.any():
                ax2.fill_between(x, ys - std, ys + std,
                                 color=c, alpha=0.10, linewidth=0, zorder=2)
        line, = ax2.plot(
            x, ys,
            color=c,
            linestyle=variant_linestyle(static_variant),
            marker=variant_marker(static_variant),
            markersize=5, markeredgecolor="white", markeredgewidth=0.4,
            linewidth=1.2, alpha=0.85, zorder=3,
        )
        ax2.set_ylim(0.5, 1.02)
        ax2.set_ylabel(f"accept rate ({sa.label_for(static_variant)})",
                       color=variant_color(static_variant), fontsize=10)
        ax2.tick_params(axis="y", labelcolor=variant_color(static_variant),
                        labelsize=10)
        ax2.spines["top"].set_visible(False)
        handles.append(line)
        labels.append(sa.label_for(static_variant))

    ax.set_xticks(x)
    ax.set_xticklabels(blocks, rotation=40, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0.0, ymax)
    ax.set_ylabel("Accept rate", fontsize=11)
    # Intentionally NO horizontal y-grid -- the dotted random reference
    # lines and (when present) the static-twin axis horizontals already
    # carry meaning, and adding faint gridlines on the same orientation
    # makes them indistinguishable from the references.
    ax.grid(False, axis="y")
    ax.grid(False, axis="x")
    if title:
        ax.set_title(title, fontsize=11, loc="left")

    # Combined legend: filter polylines + dashed random references +
    # static (if drawn on a twinned axis).  Putting the random refs
    # into the legend instead of as right-edge text avoids collision
    # with the static axis tick labels.
    legend_handles = list(handles)
    legend_labels = list(labels)
    legend_handles.extend(random_handles)
    legend_labels.extend(h.get_label() for h in random_handles)
    if legend_handles:
        # Park the legend below the figure (well clear of the rotated
        # x-tick labels and the y-axis tick numbers), with the bottom
        # padding reserved by tight_layout's `rect`.
        fig.legend(
            legend_handles, legend_labels,
            fontsize=11, loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(legend_handles), 5),
            framealpha=0.9, columnspacing=1.5, handlelength=1.8,
        )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig, ax


def plot_per_category_routing(
    rate_grids: Mapping[str, pd.DataFrame],
    *,
    filter_variants: Sequence[str],
    panel_titles: Optional[Mapping[str, str]] = None,
    bucket_labels: Optional[Mapping[str, Mapping[str, str]]] = None,
    random_ref: Optional[Tuple[str, float]] = None,
    n_cols: int = 2,
    figsize_per_panel: Tuple[float, float] = (4.5, 3.4),
    title: str = "",
    ymax: Optional[float] = None,
) -> Tuple[Figure, np.ndarray]:
    """Per-category accept-rate plot, one small-multiples panel per category.

    Each panel is a line plot in the style of `plot_per_block_routing_lines`:
    one polyline per filter variant, x-axis is the category's bucket
    list (e.g. ``day, twilight, night``), y-axis is the empirical
    accept rate.  Reads naturally as "for buckets the filter judges
    novel, the polyline lifts up".

    Args:
        rate_grids: ``{category_name: wide_df}`` from
            :func:`streaming.per_category_routing`.  Each wide_df is
            indexed by bucket and has one column per variant label.
        filter_variants: variants whose polylines to draw, in legend
            order (the corresponding column in each ``wide_df`` is
            looked up via :func:`streaming.label_for`).
        panel_titles: optional ``{category_name: panel_title}`` map.
        bucket_labels: optional ``{category_name: {bucket: pretty_label}}``
            for x-tick relabeling (e.g. ``"rain_wet"`` -> ``"rain/wet"``).
        ymax: shared upper y-limit across panels; auto-derived from data
            when omitted.
    """
    cats = list(rate_grids.keys())
    n = len(cats)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, np.array([[ax]])

    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols,
                 figsize_per_panel[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    if ymax is None:
        ymax_data = max(
            (float(df.to_numpy().max())
             for df in rate_grids.values() if not df.empty),
            default=0.6,
        )
        ymax = max(0.6, min(1.0, ymax_data + 0.05))

    handles_for_legend: List = []
    labels_for_legend: List[str] = []
    seen_variants: set = set()
    random_ref_seen = False

    for idx, cat in enumerate(cats):
        ax = axes_flat[idx]
        grid = rate_grids[cat]
        if grid.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        buckets = list(grid.index)
        x = np.arange(len(buckets))

        # Random accept rate is uniform across buckets by construction:
        # one horizontal line per panel, drawn first so filter polylines
        # sit on top.
        if random_ref is not None:
            ref_label, ref_value = random_ref
            ref_line = ax.axhline(
                ref_value, color="#7a7a7a",
                linestyle=":", linewidth=1.3, alpha=0.85,
            )
            if not random_ref_seen:
                handles_for_legend.append(ref_line)
                labels_for_legend.append(ref_label)
                random_ref_seen = True

        for v in filter_variants:
            col = sa.label_for(v)
            if col not in grid.columns:
                continue
            ys = grid[col].values
            line, = ax.plot(
                x, ys,
                color=variant_color(v),
                linestyle=variant_linestyle(v),
                marker=variant_marker(v),
                markersize=6, markeredgecolor="white", markeredgewidth=0.4,
                linewidth=1.6, alpha=0.95,
            )
            if v not in seen_variants:
                handles_for_legend.append(line)
                labels_for_legend.append(sa.label_for(v))
                seen_variants.add(v)
        # Bucket relabeling for tick text only (data column lookup
        # uses the canonical bucket name from the grid index).
        xtick_labels = [
            (bucket_labels or {}).get(cat, {}).get(b, b)
            for b in buckets
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=10)
        ax.set_ylim(0.0, ymax)
        ax.set_ylabel("Accept rate", fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.grid(False, axis="x")
        ttl = (panel_titles or {}).get(cat, cat.replace("_", " "))
        ax.set_title(ttl, fontsize=12, loc="left")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    if handles_for_legend:
        fig.legend(
            handles_for_legend, labels_for_legend,
            loc="lower center", fontsize=10, framealpha=0.92,
            ncol=min(len(handles_for_legend), 4),
            bbox_to_anchor=(0.5, -0.04),
        )
    if title:
        fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1.0, 0.97 if title else 1.0))
    return fig, axes


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
                fontsize=12, fontweight="bold", color="#444444", alpha=0.9,
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
    title: str = "",
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
        ls = variant_linestyle(variant)
        c = variant_color(variant)
        ax.plot(xs, means, color=c, lw=1.5, alpha=0.95, ls=ls,
                label=sa.label_for(variant), zorder=3)
        # Only the solid-line variant in a (single-ref, two-ref) pair
        # gets a +/-1-seed-std band: the two share a family hue, so
        # drawing both bands collapses into a single oversized blob
        # the eye cannot decompose.  The two-ref polyline still tracks
        # the band's centerline accurately enough for cross-seed
        # comparison.
        if ls == "-":
            ax.fill_between(xs, means - stds, means + stds,
                            color=c, alpha=0.14,
                            linewidth=0, zorder=2)
    if not has_data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
    if boundaries:
        _draw_block_lines(ax, boundaries)
    if midpoints:
        _annotate_block_numbers(ax, midpoints)
    ax.set_ylabel("Accept rate", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(bottom=0.0)
    if title:
        ax.set_title(title, fontsize=14, loc="left", pad=14)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")
    if has_data:
        ax.legend(fontsize=12, loc="upper left", framealpha=0.92,
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
            ax.set_ylabel("Fraction", fontsize=13)
            ax.tick_params(axis="both", labelsize=12)
            ax.set_ylim(0, 1)
            ttl = (composition_titles or {}).get(field, f"{field} composition")
            ax.set_title(ttl, fontsize=14, loc="left")
            ax.legend(fontsize=11, loc="upper left", framealpha=0.92,
                      ncol=min(4, max(1, len(frac_df.columns))))
            ax.grid(False)

    axes[-1].set_xlabel("Frame index (post-bootstrap)", fontsize=13)
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
    title: str = "",
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
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_per_class_heatmap(
    grid: pd.DataFrame,
    *,
    title: str = "",
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
    title: str = "",
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
            label = ("Frame index (post-bootstrap)"
                     if x_col == "items_processed"
                     else x_col.replace("_", " "))
            ax.set_xlabel(label, fontsize=8)
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7.5,
                   framealpha=0.9, ncol=1, bbox_to_anchor=(1.02, 1.0))
    if title:
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
    title: str = "",
) -> Tuple[Figure, Axes]:
    """Line plot of accept rate vs refresh index for each variant."""
    fig, ax = plt.subplots(figsize=figsize)
    if seg.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    for variant, grp in seg.groupby("variant", sort=False):
        grp = grp.sort_values("refresh_idx")
        ax.plot(grp["refresh_idx"], grp["accept_rate"],
                marker=variant_marker(variant),
                linestyle=variant_linestyle(variant),
                markersize=4, linewidth=1.2,
                color=variant_color(variant),
                label=sa.label_for(variant), alpha=0.9)
    ax.set_xlabel("refresh index (-1 = pre-first-refresh)")
    ax.set_ylabel("segment accept rate")
    if title:
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
    title: str = "",
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
    if title:
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
    keep_variants = [
        "no_filter_cityday_curated",
        "random_p33_cityday_curated",
        "random_p77_cityday_curated",
        "static_p15_cityday_curated",
        "static_p20_cityday_curated",
        "adaptive_window_p20_cityday_curated",
        "adaptive_window_p20_twoRef_cityday_curated",
        "adaptive_reservoir_p20_cityday_curated",
        "adaptive_reservoir_p20_twoRef_cityday_curated",
    ]
    sub = inv[(inv["manifest"] == manifest) &
              inv["variant"].isin(keep_variants)].copy()
    sub["order"] = sub["variant"].map({v: i for i, v in enumerate(keep_variants)})
    sub = sub.sort_values("order")
    fig, ax = plt.subplots(figsize=figsize)
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    colors = [variant_color(v) for v in sub["variant"]]
    # Hatch the two-ref / non-headline variants so single-ref vs
    # two-ref pairs sharing a family hue are visually distinct in
    # greyscale print.
    hatches = ["//" if "twoRef" in v else "" for v in sub["variant"]]
    bars = ax.bar(sub["label"], sub["smoothed_mAP"], color=colors,
                  yerr=sub["smoothed_std"], capsize=2,
                  edgecolor="white", linewidth=0.5)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
            bar.set_edgecolor("white")
    for bar, ar in zip(bars, sub["accept_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{ar:.2f}", ha="center", fontsize=7, color="#333")
    ax.set_ylabel("Smoothed tail-5 mAP")
    ax.set_xlabel("(numbers above bars: empirical accept rate)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    return fig, ax
