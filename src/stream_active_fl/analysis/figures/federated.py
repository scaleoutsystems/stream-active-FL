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
        plot_stream_composition_with_partitions(composition, ...)
            stream-evolution composition panels with client partition markers

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

# Single-hue-per-family palette (matches the streaming chapter).  Hue
# encodes the family; linestyle + marker carry flavor within a family.
FAMILY_COLORS: Dict[str, str] = {
    "none":       "#2ca02c",
    "no_filter":  "#2ca02c",
    "random":     "#7f7f7f",
    "static":     "#1f77b4",
    "window":     "#ff7f0e",
    "reservoir":  "#d62728",
}

# Random baselines are distinguished from each other by lightness, not
# hue; lower accept fractions get lighter shades.
_RANDOM_GREYSCALE: Dict[str, str] = {
    "fed_random_p7_cityday_curated":              "#dcdcdc",
    "fed_random_p11_cityday_curated":             "#c4c4c4",
    "fed_random_p12_cityday_curated":             "#b4b4b4",
    "fed_random_p15_cityday_curated":             "#9c9c9c",
    "fed_random_p18_cityday_curated":             "#7f7f7f",
    "fed_random_p26_cityday_curated":             "#5b5b5b",
    "fed_random_p30_cityday_curated":             "#454545",
    "fed_random_p77_cityday_curated":             "#202020",
    "fed_random_p7_cityday_temporal":             "#dcdcdc",
    "fed_random_p11_cityday_temporal":            "#bcbcbc",
    "fed_random_p15_cityday_temporal":            "#9c9c9c",
    "fed_random_p20_cityday_temporal":            "#787878",
    "fed_random_p26_cityday_temporal":            "#5b5b5b",
    "fed_random_p7_cityday_curated_heavyLocal":   "#dcdcdc",
    "fed_random_p11_cityday_curated_heavyLocal":  "#bcbcbc",
    "fed_random_p15_cityday_curated_heavyLocal":  "#9c9c9c",
    "fed_random_p20_cityday_curated_heavyLocal":  "#787878",
    "fed_random_p26_cityday_curated_heavyLocal":  "#5b5b5b",
}


# Per-client palette for the federated 4-client `domain_aligned` setup.
CLIENT_COLORS: Dict[int, str] = {
    0: "#1f77b4",   # familiar - blue
    1: "#2ca02c",   # city-novel - green
    2: "#ff7f0e",   # urban arterial - orange
    3: "#d62728",   # out-of-city - red
}


def variant_color(variant: str, *, project_root: Optional[Path] = None) -> str:
    """Return the family-level color for a variant.

    All members of a family share the same hue; flavor is conveyed by
    linestyle + marker.  Random baselines are graded by lightness so
    iso-accept comparisons remain readable when several appear on a
    single axis.
    """
    if variant in _RANDOM_GREYSCALE:
        return _RANDOM_GREYSCALE[variant]
    fam = fa.family_for_variant(variant, project_root=project_root)
    return FAMILY_COLORS.get(fam, "#444444")


def variant_linestyle(variant: str) -> str:
    """Return a stable matplotlib linestyle encoding variant flavor.

    - solid ``-``   : single-reference / vanilla / random / no-filter
    - dashed ``--`` : two-reference Mahalanobis variant
    - dotted ``:``  : noBoot ablation, the sparse-refresh diagnostic
    - dash-dot ``-.``: heavy-local schedule (FL-only)
    """
    if "twoRef" in variant:
        return "--"
    if "noBoot" in variant:
        return ":"
    if "sparseRefresh" in variant:
        return ":"
    if variant.endswith("_heavyLocal"):
        return "-."
    return "-"


def variant_marker(variant: str) -> str:
    """Return a stable marker shape encoding the variant flavor.

    - ``o``: vanilla / single-ref (default)
    - ``s``: two-reference Mahalanobis
    - ``X``: noBoot / sparse-refresh diagnostic
    - ``D``: static threshold filter
    - ``*``: no-filter ceiling
    - ``.``: random baseline
    """
    if "twoRef" in variant:
        return "s"
    if "noBoot" in variant or "sparseRefresh" in variant:
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
    headline_variants: Optional[Sequence[str]] = None,
    annotate_filters_only: bool = True,
    figsize: Tuple[float, float] = (7.0, 4.6),
) -> Tuple[Figure, Axes]:
    """Single-cell iso-accept scatter (mirrors streaming `plot_inventory_scatter`).

    Plots smoothed tail-K mAP vs effective accept rate for one
    ``(manifest, schedule)`` cell.  All randoms in the cell are used
    to draw the iso-accept envelope (dotted grey curve + small grey
    markers with seed-std error bars); ``headline_variants`` restricts
    the colored *filter* points to a chosen subset.  Marker labels use
    the inventory's short label with the per-cell parenthetical
    ("(heavy-local)", "(temporal)", ...) stripped, since the figure
    caption already names the cell.

    Args:
        inv: Inventory DataFrame from `inventory_table`.
        manifest: ``"curated"`` or ``"temporal"``.
        schedule: ``"default"`` or ``"heavyLocal"``.
        headline_variants: Optional whitelist of non-random variants
            to render as colored markers; randoms are always drawn as
            the envelope regardless of this list.
        annotate_filters_only: If True, the no-filter star and random
            dots get no per-point label (the family legend and dotted
            envelope are self-explanatory).
        figsize: ``(w, h)`` figure size.

    Returns:
        ``(fig, ax)`` for the rendered scatter.
    """
    sub = inv[(inv["manifest"] == manifest)
              & (inv["schedule"] == schedule)].copy()
    sub = sub.dropna(subset=["accept_rate", "smoothed_mAP"])
    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)

    # Random envelope: dotted polyline + small grey markers with
    # vertical std bars so the envelope visibly decomposes into
    # measured points instead of looking like an interpolation.
    rand = sub[sub["family"] == "random"].sort_values("accept_rate")
    rand_color = FAMILY_COLORS.get("random", "#7f7f7f")
    if not rand.empty:
        ax.plot(rand["accept_rate"], rand["smoothed_mAP"],
                color=rand_color, linestyle=":", linewidth=1.2,
                alpha=0.8, label="iso-accept random envelope", zorder=1)
        for _, row in rand.iterrows():
            std = row["smoothed_std"] if pd.notna(row["smoothed_std"]) else 0
            ax.errorbar(
                row["accept_rate"], row["smoothed_mAP"], yerr=std,
                fmt=".", color=rand_color, markersize=5, capsize=2,
                elinewidth=0.6, alpha=0.7, zorder=2,
            )

    if headline_variants is not None:
        sub_pts = sub[sub["variant"].isin(list(headline_variants))]
    else:
        sub_pts = sub
    sub_pts = sub_pts[sub_pts["family"] != "random"]

    # The figure caption identifies the cell, so labels do not need to
    # repeat "(heavy-local)" / "(temporal)" annotations.
    strip_tags: List[str] = []
    if schedule == "heavyLocal" and manifest == "temporal":
        strip_tags.extend([" (temporal, heavy-local)", "temporal, heavy-local, ",
                           ", temporal, heavy-local", "temporal, heavy-local"])
    if schedule == "heavyLocal":
        strip_tags.extend([" (heavy-local)", "heavy-local, ",
                           ", heavy-local", "heavy-local"])
    if manifest == "temporal":
        strip_tags.extend([" (temporal)", "temporal, ", ", temporal", "temporal"])

    def _short_label(text: str) -> str:
        out = text
        for tag in strip_tags:
            out = out.replace(tag, "")
        out = out.replace("(, ", "(").replace(", )", ")")
        out = out.replace("()", "").strip()
        return out

    for _, row in sub_pts.iterrows():
        std = row["smoothed_std"] if pd.notna(row["smoothed_std"]) else 0
        ax.errorbar(
            row["accept_rate"], row["smoothed_mAP"], yerr=std,
            fmt=variant_marker(row["variant"]),
            color=variant_color(row["variant"]),
            markersize=9, markeredgecolor="white", markeredgewidth=0.6,
            capsize=2, elinewidth=0.6, alpha=0.95, zorder=3,
        )
        skip = annotate_filters_only and row["family"] in {"random", "none"}
        if not skip:
            ax.annotate(
                _short_label(row["label"]),
                (row["accept_rate"], row["smoothed_mAP"]),
                xytext=(6, 5), textcoords="offset points",
                fontsize=9, color="#333333",
            )

    ax.set_xlabel("Effective accept rate", fontsize=11)
    ax.set_ylabel("Smoothed tail-5 mAP", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

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
                    xytext=(5, 4), textcoords="offset points", fontsize=8.5)

    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Filter accept rate", fontsize=11)
    ax.set_ylabel(r"$\Delta$ smoothed mAP (filter $-$ iso-accept random)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_iso_accept_leaderboard_grid(
    inv: pd.DataFrame,
    *,
    cells: Optional[Sequence[Tuple[str, str, str]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    annotate: bool = True,
    annotate_variants: Optional[Sequence[str]] = None,
    restrict_to_variants: Optional[Sequence[str]] = None,
) -> Tuple[Figure, np.ndarray]:
    """Federated analogue of the streaming iso-accept leaderboard.

    One panel per (schedule, partition) cell.  Each panel plots
    smoothed tail-K mAP vs effective accept rate for every variant in
    that cell.  Random points appear as discrete grey markers with
    error bars and a dotted envelope through them; filter variants are
    colored by family with their family-mean error bar.  A no-filter
    ceiling (when present) anchors the right edge.

    Args:
        inv: Inventory DataFrame from `inventory_table` with at least
            `variant`, `manifest`, `schedule`, `family`, `accept_rate`,
            `smoothed_mAP`, `smoothed_std`, and `label` columns.
        cells: Sequence of `(panel_title, manifest, schedule)` tuples
            in the order they should appear (row-major).  Defaults to
            the four cells of the chapter's 2 x 2 design.
        figsize: Optional ``(w, h)`` override.
        ymin, ymax: Optional shared y-axis limits.  When omitted, each
            panel uses its own data extent with a small padding.
        annotate: If True, label each non-random marker with its short
            variant label.
        annotate_variants: When given, annotate only these variants.
            Overrides ``annotate``.  Use to restrict labels to the
            headline filters in panels that would otherwise overplot.
        restrict_to_variants: When given, only plot these variants
            (the random envelope is still computed from every random
            run in the cell).

    Returns:
        ``(fig, axes)`` where ``axes`` is the 2D array of panel axes.
    """
    if cells is None:
        cells = (
            ("Default schedule, curated partition",      "curated",  "default"),
            ("Default schedule, temporal partition",     "temporal", "default"),
            ("Heavy-local schedule, curated partition",  "curated",  "heavyLocal"),
            ("Heavy-local schedule, temporal partition", "temporal", "heavyLocal"),
        )
    n = len(cells)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (6.5 * n_cols, 4.6 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, (title, manifest, schedule) in enumerate(cells):
        ax = axes_flat[idx]
        sub = inv[(inv["manifest"] == manifest)
                  & (inv["schedule"] == schedule)].copy()
        sub = sub.dropna(subset=["accept_rate", "smoothed_mAP"])
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=12, loc="left")
            continue

        # Random envelope + discrete markers with seed-std error bars.
        rand = sub[sub["family"] == "random"].sort_values("accept_rate")
        rand_color = FAMILY_COLORS.get("random", "#7f7f7f")
        if not rand.empty:
            ax.plot(
                rand["accept_rate"], rand["smoothed_mAP"],
                color=rand_color, linestyle=":", linewidth=1.4,
                alpha=0.85, label="iso-accept random envelope", zorder=1,
            )
            for _, row in rand.iterrows():
                std = row["smoothed_std"] if pd.notna(row["smoothed_std"]) else 0
                ax.errorbar(
                    row["accept_rate"], row["smoothed_mAP"], yerr=std,
                    fmt=".", color=rand_color, markersize=5, capsize=2,
                    elinewidth=0.6, alpha=0.7, zorder=2,
                )

        # Non-random variants: color by family, marker by flavor.
        non_rand = sub[sub["family"] != "random"]
        if restrict_to_variants is not None:
            keep = set(restrict_to_variants)
            non_rand = non_rand[non_rand["variant"].isin(keep)]
        annot_set = (set(annotate_variants)
                     if annotate_variants is not None else None)
        # Strip the "(heavy-local)" / "(temporal, ...)" parenthetical
        # from labels in panels whose title already conveys that
        # information.  We try the longer (combined) tags first so the
        # final fallback patterns do not eat their own context.
        strip_tags = []
        if schedule == "heavyLocal" and manifest == "temporal":
            strip_tags.extend(["temporal, heavy-local, ",
                               "(temporal, heavy-local)",
                               "temporal, heavy-local"])
        if schedule == "heavyLocal":
            strip_tags.extend([" (heavy-local)", "heavy-local, ",
                               ", heavy-local", "heavy-local"])
        if manifest == "temporal":
            strip_tags.extend([" (temporal)", "temporal, ",
                               ", temporal", "temporal"])

        def _short_label(text: str) -> str:
            out = text
            for tag in strip_tags:
                out = out.replace(tag, "")
            # Tidy leftover "(, "/", )"/"()" artifacts after stripping.
            out = out.replace("(, ", "(").replace(", )", ")")
            out = out.replace("()", "").strip()
            return out

        # Sort annotated rows by mAP so the alternating offset is
        # deterministic and tight clusters get split above/below.
        annot_rows = [(i, row) for i, row in non_rand.iterrows()
                      if (annot_set is None and annotate)
                      or (annot_set is not None and row["variant"] in annot_set)]
        annot_rows.sort(key=lambda kv: (kv[1]["accept_rate"],
                                        kv[1]["smoothed_mAP"]))
        for _, row in non_rand.iterrows():
            std = row["smoothed_std"] if pd.notna(row["smoothed_std"]) else 0
            ax.errorbar(
                row["accept_rate"], row["smoothed_mAP"], yerr=std,
                fmt=variant_marker(row["variant"]),
                color=variant_color(row["variant"]),
                markersize=10, markeredgecolor="white", markeredgewidth=0.7,
                capsize=2, elinewidth=0.7, alpha=0.95, zorder=3,
            )
        for k, (_, row) in enumerate(annot_rows):
            # Alternate offset direction (NE / SE) so adjacent
            # annotations do not stack on top of each other; use a
            # generous magnitude so labels stay legible even when
            # adjacent markers cluster very tightly.
            dy = 10 if (k % 2 == 0) else -16
            ax.annotate(
                _short_label(row["label"]),
                (row["accept_rate"], row["smoothed_mAP"]),
                xytext=(8, dy), textcoords="offset points",
                fontsize=10, color="#222222",
            )

        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlabel("Effective accept rate", fontsize=11)
        ax.set_ylabel("Smoothed tail-5 mAP", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    # One shared family legend at the figure bottom (drops random
    # marker since it already has its own envelope label and the grey
    # dot conflates with axis grid).
    fams_in_plot = sorted(
        {f for f in inv["family"].unique() if f != "random"})
    handles = [mpatches.Patch(color=FAMILY_COLORS.get(f, "#444"), label=f)
               for f in fams_in_plot]
    if handles:
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
                   fontsize=11, framealpha=0.9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig, axes


def plot_overall_mAP_trajectory(
    traj: pd.DataFrame,
    *,
    x_col: str = "items_processed_total",
    figsize: Tuple[float, float] = (9.0, 5.0),
    title: str = "",
    smoothing_window: int = 1,
    show_std: bool = True,
    show_std_for: Optional[Sequence[str]] = None,
    xlabel: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot overall mAP vs ``x_col`` for multiple variants (already aggregated).

    Args:
        traj: Long-format trajectory from `federated.mAP_trajectory`.
        x_col: X-axis column (``items_processed_total``, ``round``, ...).
        figsize: ``(w, h)`` figure size.
        title: Optional panel title.
        smoothing_window: Centered rolling-mean window applied to each
            variant's trace.  Set to ``1`` to disable.
        show_std: If True and ``mAP_std`` is present, draws a seed-std
            band around each variant.
        show_std_for: Optional whitelist of variants that get a seed-std
            band; when provided, takes precedence over ``show_std`` and
            band is drawn only for the listed variants (use to keep
            cluttered legends readable).
        xlabel: Override the x-axis label.  Defaults to a sensible
            mapping from ``x_col`` (e.g. "Communication round" for
            ``"round"``).
    """
    fig, ax = plt.subplots(figsize=figsize)
    if traj.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    if x_col not in traj.columns:
        x_col = "round"
    band_set = set(show_std_for) if show_std_for is not None else None
    for v, grp in traj.groupby("variant", sort=False):
        grp = grp.sort_values(x_col)
        c = variant_color(v)
        ls = variant_linestyle(v)
        y = grp["mAP"].astype(float)
        if smoothing_window and smoothing_window > 1:
            y = y.rolling(smoothing_window, center=True,
                          min_periods=1).mean()
        ax.plot(grp[x_col], y, color=c, linestyle=ls,
                label=fa.label_for(v), linewidth=1.8, alpha=0.92)
        wants_band = (band_set is None and show_std) or (
            band_set is not None and v in band_set)
        if wants_band and "mAP_std" in grp.columns:
            std = grp["mAP_std"].fillna(0).astype(float)
            if smoothing_window and smoothing_window > 1:
                std = std.rolling(smoothing_window, center=True,
                                  min_periods=1).mean()
            ax.fill_between(grp[x_col], y - std, y + std,
                            color=c, alpha=0.13, linewidth=0)

    default_xlabel = {"items_processed_total": "Items processed (cumulative)",
                      "optimizer_steps_total": "Optimizer steps (cumulative)",
                      "round": "Communication round"}.get(x_col, x_col)
    ax.set_xlabel(xlabel or default_xlabel, fontsize=12)
    ax.set_ylabel("Validation mAP", fontsize=12)
    ax.tick_params(labelsize=11)
    if title:
        ax.set_title(title, fontsize=13, loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")
    ax.legend(loc="lower right", fontsize=10, ncol=2, framealpha=0.9)
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
    title: str = "",
    ymax: Optional[float] = None,
    label_map: Optional[Mapping[str, str]] = None,
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
        figsize = (max(9.0, 1.4 * n_var + 2.5), 5.4)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_var)
    bar_w = 0.85 / max(1, n_clients)

    # Pick the right per-client label dict based on the variants shown.
    label_dict = fa.CLIENT_LABEL
    if any("_temporal" in v for v in variant_order) and not any(
            "_curated" in v for v in variant_order):
        label_dict = fa.TEMPORAL_CLIENT_LABEL

    for cid in range(n_clients):
        sub = df[df["client"] == cid].set_index("variant").reindex(variant_order)
        ax.bar(
            x + cid * bar_w - 0.425 + bar_w / 2,
            sub["accept_rate"].values,
            yerr=sub["accept_rate_std"].fillna(0).values,
            width=bar_w,
            color=CLIENT_COLORS.get(cid, "#888888"),
            label=label_dict.get(cid, f"C{cid}"),
            edgecolor="white", linewidth=0.5,
            capsize=2,
        )

    def _xlabel(v: str) -> str:
        if label_map is not None and v in label_map:
            return label_map[v]
        return fa.label_for(v)

    ax.set_xticks(x)
    ax.set_xticklabels([_xlabel(v) for v in variant_order],
                       rotation=25, ha="right", fontsize=11)
    ax.set_ylabel("Accept rate", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    if title:
        ax.set_title(title, fontsize=13, loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=11, ncol=n_clients,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return fig, ax


def plot_novelty_routing(
    novelty: pd.DataFrame,
    *,
    manifest: str = "curated",
    schedule: str = "default",
    variants: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "",
    annotate_values: bool = True,
) -> Tuple[Figure, Axes]:
    """Horizontal bar chart of novelty_ratio per variant.

    Args:
        novelty: Output of `federated.novelty_routing_summary`.
        manifest: Restrict to this manifest (``"curated"`` /
            ``"temporal"``).
        schedule: Restrict to this schedule (``"default"`` /
            ``"heavyLocal"``).
        variants: Optional whitelist of variants to include (preserves
            order).  When omitted, every variant matching the
            (manifest, schedule) filter is plotted.
        figsize: Optional ``(w, h)`` override.
        title: Optional panel title.
        annotate_values: If True, prints the numeric ratio next to each
            bar so the figure is readable without consulting the table.

    Bars > 1 mean the variant routes more compute to novel-domain
    clients than to the familiar one; bars ~ 1 mean flat (random-like).
    """
    if "schedule" in novelty.columns:
        sub = novelty[novelty["schedule"] == schedule].copy()
    else:
        sub = novelty.copy()
    sub = sub[sub["variant"].apply(fa.manifest_for_variant) == manifest]
    sub = sub.dropna(subset=["novelty_ratio"])
    if variants is not None:
        sub = sub[sub["variant"].isin(list(variants))]
        # Preserve caller order rather than sorting by ratio.
        sub = sub.set_index("variant").reindex(
            [v for v in variants if v in sub.set_index("variant").index]
        ).reset_index()
    else:
        sub = sub.sort_values("novelty_ratio")
    if figsize is None:
        figsize = (8.0, max(3.0, 0.45 * len(sub) + 1.4))
    fig, ax = plt.subplots(figsize=figsize)
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return fig, ax
    colors = [variant_color(v) for v in sub["variant"]]
    bars = ax.barh(sub["label"], sub["novelty_ratio"], color=colors,
                   edgecolor="white", linewidth=0.6)
    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1.0,
               alpha=0.75)
    ax.set_xlabel(
        r"Novelty ratio = mean(C1, C2, C3 accept) / C0 accept",
        fontsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    if annotate_values:
        xmax = float(sub["novelty_ratio"].max())
        offset = max(0.02, 0.02 * xmax)
        for bar, val in zip(bars, sub["novelty_ratio"]):
            ax.text(bar.get_width() + offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=10,
                    color="#222222")
        ax.set_xlim(0, max(sub["novelty_ratio"].max() * 1.18, 1.25))
    if title:
        ax.set_title(title, fontsize=13, loc="left")
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
    title: str = r"Per-block $\Delta$mAP (filter $-$ iso-accept random)",
    figsize: Optional[Tuple[float, float]] = None,
    family_separator: bool = True,
    column_rename: Optional[Mapping[str, str]] = None,
) -> Tuple[Figure, Axes]:
    """Diverging heatmap of (block x pair) Δ-mAP, centered on 0.

    Args:
        delta_grid: Block-by-pair grid of mAP deltas.
        vlim: Optional symmetric color limit (defaults to data extent).
        title: Title text for the panel.
        figsize: Optional ``(w, h)`` override.
        family_separator: If True the rows are reordered with familiar
            blocks at the top and novel blocks below, with a horizontal
            line drawn at the family boundary.
        column_rename: Optional ``{old_label: new_label}`` mapping for
            the column headers.  Use this to swap full filter+random
            labels for short cell tags ("default+curated", etc.) so
            the heatmap is legible in print without rotating the text.
    """
    if delta_grid.empty:
        fig, ax = plt.subplots(figsize=figsize or (6.5, 4.0))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    df = delta_grid.copy()
    if column_rename:
        df = df.rename(columns=dict(column_rename))
    if family_separator:
        fam_idx = [b for b in df.index if fa.block_family(b) == "familiar"]
        nov_idx = [b for b in df.index if fa.block_family(b) == "novel"]
        oth_idx = [b for b in df.index if b not in fam_idx and b not in nov_idx]
        df = df.loc[fam_idx + nov_idx + oth_idx]
        boundary = len(fam_idx)
    else:
        boundary = None

    if figsize is None:
        figsize = (max(6.5, 1.3 * df.shape[1] + 2.0),
                   max(3.8, 0.36 * df.shape[0] + 1.6))
    fig, ax = plt.subplots(figsize=figsize)
    if vlim is None:
        v = float(df.abs().to_numpy().max())
        vlim = v if v > 0 else 0.01
    im = _heatmap(df, ax=ax, cmap="RdBu_r",
                  vmin=-vlim, vmax=+vlim, title=title,
                  annotate=True, fmt="{:+.3f}",
                  xtick_fontsize=11, ytick_fontsize=11,
                  cell_fontsize=10, xtick_rotation=20)
    if boundary and 0 < boundary < df.shape[0]:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.0, alpha=0.8)
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
    x_col: str = "checkpoint_idx",
    n_cols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4.0, 2.6),
    title: str = "",
    smoothing_window: int = 1,
    shade_familiar: bool = True,
    show_std: bool = False,
    title_suffix: Optional[str] = None,
    xlabel: Optional[str] = None,
) -> Tuple[Figure, np.ndarray]:
    """Per-block mAP-over-time, faceted with one panel per block.

    Args:
        trajectories: ``{variant_name: long_df}`` from
            `federated.per_block_trajectory`.
        blocks: Ordered list of validation blocks (one per panel).
        x_col: Column to use as x-axis (typically
            ``checkpoint_idx`` or ``round``).
        n_cols: Number of panel columns.
        figsize_per_panel: ``(w, h)`` per panel.
        title: Optional figure title.
        smoothing_window: Centered rolling-mean window applied to each
            variant's trace.  Set to ``1`` to disable.
        shade_familiar: Tint the panel background for familiar blocks.
        show_std: If True and ``mAP_std`` exists, draws a seed-std
            band around each variant.
        title_suffix: Override the default ``"({family})"`` panel-title
            suffix (e.g. set to ``""`` for per-category panels where
            every bucket is novel by construction and the suffix is
            redundant).  ``None`` keeps the default behavior.
        xlabel: Override the bottom-row x-axis label.  Defaults to
            ``x_col`` with underscores converted to spaces.
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
            y = sub["mAP"].astype(float)
            if smoothing_window and smoothing_window > 1:
                y = y.rolling(smoothing_window, center=True,
                              min_periods=1).mean()
            ax.plot(sub[x_col], y,
                    color=variant_color(variant),
                    linestyle=variant_linestyle(variant),
                    label=fa.label_for(variant), linewidth=1.6, alpha=0.92)
            if show_std and "mAP_std" in sub.columns:
                std = sub["mAP_std"].fillna(0).astype(float)
                if smoothing_window and smoothing_window > 1:
                    std = std.rolling(smoothing_window, center=True,
                                      min_periods=1).mean()
                ax.fill_between(sub[x_col], y - std, y + std,
                                color=variant_color(variant),
                                alpha=0.12, linewidth=0)
        family = fa.block_family(block)
        if shade_familiar and family == "familiar":
            ax.set_facecolor("#fff7e6")
        if title_suffix is None:
            panel_title = f"{block} ({family})"
        elif title_suffix:
            panel_title = f"{block} {title_suffix}"
        else:
            panel_title = block
        ax.set_title(panel_title, fontsize=11, loc="left")
        ax.grid(True, axis="y", alpha=0.3)
        ax.grid(False, axis="x")
        ax.set_ylabel("mAP", fontsize=10)
        ax.tick_params(labelsize=10)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(xlabel or x_col.replace("_", " "), fontsize=10)
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", fontsize=11,
                   framealpha=0.9,
                   ncol=min(len(handles), 4),
                   bbox_to_anchor=(0.5, 0.0))
    if title:
        fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    bottom_pad = 0.07 if handles else 0.02
    fig.tight_layout(rect=(0, bottom_pad, 1.0, 0.98 if title else 1.0))
    return fig, axes


def plot_per_block_trajectory_delta(
    traj_delta: pd.DataFrame,
    *,
    metric: str = "cum_avg_delta",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    column_rename: Optional[Mapping[str, str]] = None,
    column_order: Optional[Sequence[str]] = None,
) -> Tuple[Figure, Axes]:
    """Per-block ``cum_avg_delta`` (or ``final_delta``) for each iso-accept pair.

    ``traj_delta`` is the long-format DataFrame from
    `federated.per_block_trajectory_delta`.  Rows are blocks; columns
    are pairings.  Familiar blocks are pinned at the top and separated
    from novel blocks by a horizontal line.

    Args:
        traj_delta: Long-format trajectory delta from
            `federated.per_block_trajectory_delta`.
        metric: Which numeric column to pivot on.
        figsize: Optional ``(w, h)`` override.
        title: Optional explicit title; falls back to ``"Per-block <metric>"``.
        column_rename: Optional ``{old_label: new_label}`` mapping for
            cleaner column headers (e.g. cell tags instead of full
            filter+random labels).
        column_order: Optional explicit column order applied AFTER the
            rename; columns not present in ``traj_delta`` are dropped.
    """
    if traj_delta.empty:
        fig, ax = plt.subplots(figsize=figsize or (6.5, 4.0))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    pivot = traj_delta.pivot_table(
        index="block", columns="filter_label", values=metric,
        aggfunc="mean",
    )
    if column_rename:
        pivot = pivot.rename(columns=dict(column_rename))
    if column_order:
        cols = [c for c in column_order if c in pivot.columns]
        if cols:
            pivot = pivot[cols]
    fam_idx = [b for b in pivot.index if fa.block_family(b) == "familiar"]
    nov_idx = [b for b in pivot.index if fa.block_family(b) == "novel"]
    oth_idx = [b for b in pivot.index if b not in fam_idx and b not in nov_idx]
    pivot = pivot.loc[fam_idx + nov_idx + oth_idx]
    boundary = len(fam_idx)

    if figsize is None:
        figsize = (max(6.5, 1.3 * pivot.shape[1] + 2.0),
                   max(3.8, 0.36 * pivot.shape[0] + 1.6))
    fig, ax = plt.subplots(figsize=figsize)
    v = float(pivot.abs().to_numpy().max())
    vlim = v if v > 0 else 0.01
    title_txt = (title or f"Per-block {metric}\n(filter mAP $-$ iso-accept random mAP)")
    im = _heatmap(pivot, ax=ax, cmap="RdBu_r",
                  vmin=-vlim, vmax=+vlim, title=title_txt,
                  annotate=True, fmt="{:+.4f}",
                  xtick_fontsize=11, ytick_fontsize=11,
                  cell_fontsize=10, xtick_rotation=20)
    if 0 < boundary < pivot.shape[0]:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.0, alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=metric)
    fig.tight_layout()
    return fig, ax


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
        ax.scatter(row["balanced_mAP"], row["worst_block_mAP"],
                   s=70, color="#444444", marker="o", alpha=0.85,
                   edgecolor="white")
        ax.annotate(row["variant_label"],
                    (row["balanced_mAP"], row["worst_block_mAP"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Balanced (mean) per-block mAP")
    ax.set_ylabel("Worst-block mAP")
    if title:
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
    title: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    baseline_label: str = "baseline",
    ablated_label: str = "ablated",
    show_baseline_std: bool = True,
    ymin: Optional[float] = None,
    delta_format: str = "{:+.4f}",
) -> Tuple[Figure, Axes]:
    """Side-by-side bars of baseline vs ablated smoothed mAP per pair.

    Args:
        pair_df: DataFrame from `federated.ablation_pair_table` with at
            least ``pair``, ``baseline_smoothed``, ``ablated_smoothed``,
            ``delta_smoothed`` columns; ``baseline_smoothed_std`` and
            ``ablated_smoothed_std`` are used as error bars when
            present.
        title: Optional figure title.
        figsize: Optional ``(w, h)`` override.
        baseline_label, ablated_label: Legend labels.
        show_baseline_std: If True (default), draws seed-std error bars
            on the baseline bars when std columns are available.
        ymin: Optional explicit y-lower bound (use to zoom in when all
            bars sit in a narrow range).
        delta_format: Format string for the delta annotations.
    """
    if figsize is None:
        figsize = (max(8.0, 1.4 * len(pair_df) + 1.5), 4.6)
    fig, ax = plt.subplots(figsize=figsize)
    if pair_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    x = np.arange(len(pair_df))
    bar_w = 0.38
    base_yerr = (pair_df["baseline_smoothed_std"].fillna(0).values
                 if show_baseline_std and "baseline_smoothed_std" in pair_df.columns
                 else None)
    abl_yerr = (pair_df["ablated_smoothed_std"].fillna(0).values
                if show_baseline_std and "ablated_smoothed_std" in pair_df.columns
                else None)
    ax.bar(x - bar_w / 2, pair_df["baseline_smoothed"], width=bar_w,
           color="#4f81bd", edgecolor="white", linewidth=0.6,
           yerr=base_yerr, capsize=2, label=baseline_label)
    ax.bar(x + bar_w / 2, pair_df["ablated_smoothed"], width=bar_w,
           color="#e07b39", edgecolor="white", linewidth=0.6,
           yerr=abl_yerr, capsize=2, label=ablated_label)

    # Offset annotations above the tallest bar+error so they sit
    # cleanly clear of the seed-std caps.
    base_top = pair_df["baseline_smoothed"] + (
        pair_df["baseline_smoothed_std"].fillna(0)
        if "baseline_smoothed_std" in pair_df.columns else 0)
    abl_top = pair_df["ablated_smoothed"] + (
        pair_df["ablated_smoothed_std"].fillna(0)
        if "ablated_smoothed_std" in pair_df.columns else 0)
    bar_tops = np.maximum(base_top.values, abl_top.values)
    headroom = 0.018 * (np.nanmax(bar_tops) - (ymin if ymin is not None else 0))
    headroom = max(headroom, 0.002)
    for xi, d, top in zip(x, pair_df["delta_smoothed"], bar_tops):
        if pd.notna(d):
            color = "#0a7d28" if d >= 0 else "#a52424"
            ax.annotate(
                "Δ=" + delta_format.format(d),
                xy=(xi, top + headroom),
                ha="center", va="bottom", fontsize=10, color=color,
                fontweight="bold",
            )
    # Make room for the topmost annotation.
    ax.set_ylim(top=np.nanmax(bar_tops) + 4 * headroom + 0.002)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_df["pair"], rotation=18, ha="right",
                       fontsize=11)
    ax.set_ylabel("Smoothed tail-5 mAP", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if title:
        ax.set_title(title, fontsize=13, loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=11, framealpha=0.9, loc="lower left")
    fig.tight_layout()
    return fig, ax


# =============================================================================
# Per-client block composition (federated analogue of streaming
# composition stack)
# =============================================================================

# Stable color palette for the 4-way client-affinity grouping.  Picked
# to be readable on a stacked bar and to match the per-client palette
# (CLIENT_COLORS) where possible (familiar=blue, novel-city=green,
# urban-arterial=orange, highway+rural=red).
GROUP_COLORS: Dict[str, str] = {
    "city day":           "#1f77b4",
    "city night/twi/wet": "#2ca02c",
    "urban arterial":     "#ff7f0e",
    "highway + rural":    "#d62728",
    "unknown":            "#888888",
}


def plot_per_client_dimension_composition(
    composition: Mapping[str, pd.DataFrame],
    *,
    client_labels: Optional[Mapping[int, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "",
) -> Tuple[Figure, np.ndarray]:
    """Per-client TOD / road / weather composition bars.

    Federated analogue of streaming fig 02's lower composition panels.
    One subplot per dimension; each subplot draws a horizontal stacked
    bar per client (one bar = 100% of that client's data) using the
    canonical project palettes (`runs.TOD_COLORS`, `runs.DOMAIN_COLORS`,
    `runs.WEATHER_COLORS`).  This makes the per-client composition
    readable at a glance and consistent with the streaming chapter.

    Args:
        composition: ``{field: wide_df}`` from
            `federated.per_client_dimension_composition`.  Empty
            DataFrames are skipped.
        client_labels: ``{client_id: display_name}``.  Defaults to
            `federated.CLIENT_LABEL`.
        figsize: Optional ``(w, h)`` override.
        title: Optional figure-level title.
    """
    items = [(f, df) for f, df in composition.items()
             if df is not None and not df.empty]
    n = len(items)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize or (6, 3))
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return fig, np.array([ax])
    if figsize is None:
        # Stack the dimensions vertically (mirrors the streaming
        # composition figure) so each panel gets its own legend row
        # and there is real estate for the four client labels on the
        # y-axis.
        figsize = (10.0, 1.9 * n + 0.6)
    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False, sharex=True)
    axes_flat = axes[:, 0]

    title_for = {
        "time_of_day": "Time of day",
        "road_type":   "Road type",
        "weather":     "Weather",
    }
    palette_for = {
        "time_of_day": ah.TOD_COLORS,
        "road_type":   ah.DOMAIN_COLORS,
        "weather":     ah.WEATHER_COLORS,
    }
    short_for = {
        "time_of_day": ah.TOD_SHORT,
        "road_type":   ah.ROAD_SHORT,
        "weather":     ah.WEATHER_SHORT,
    }
    cl_label = dict(client_labels or fa.CLIENT_LABEL)

    for ax, (field, frac) in zip(axes_flat, items):
        clients = list(frac.index)
        y = np.arange(len(clients))
        offset = np.zeros(len(clients), dtype=float)
        palette = palette_for.get(field, {})
        short = short_for.get(field, {})
        legend_handles = []
        for col in frac.columns:
            widths = frac[col].to_numpy(dtype=float)
            if widths.sum() == 0:
                continue
            color = palette.get(col, None)
            ax.barh(y, widths, left=offset, color=color or "#888",
                    edgecolor="white", linewidth=0.6, alpha=0.85)
            # Annotate the largest bar of this category in white text
            # if it's wide enough to fit the short name.
            idx_largest = int(np.argmax(widths))
            if widths[idx_largest] >= 0.12:
                ax.text(offset[idx_largest] + widths[idx_largest] / 2,
                        y[idx_largest], short.get(col, col),
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            offset += widths
            legend_handles.append(mpatches.Patch(color=color or "#888",
                                                 label=short.get(col, col)))

        ax.set_yticks(y)
        ax.set_yticklabels([cl_label.get(int(c), f"C{int(c)}") for c in clients],
                           fontsize=11)
        ax.set_xlim(0, 1.0)
        ax.tick_params(axis="x", labelsize=10)
        ax.invert_yaxis()
        ax.set_title(title_for.get(field, field), fontsize=13, loc="left")
        ax.grid(True, axis="x", alpha=0.3)
        # Per-panel legend on the panel's right side so it does not
        # collide with the next panel's title or with the shared
        # bottom x-axis label.
        if legend_handles:
            ax.legend(handles=legend_handles, loc="center left",
                      bbox_to_anchor=(1.01, 0.5),
                      ncol=1,
                      fontsize=10, frameon=False, handlelength=1.4)

    axes_flat[-1].set_xlabel("Fraction of client's data", fontsize=11)
    if title:
        fig.suptitle(title, fontsize=14, x=0.01, ha="left")
    top_pad = 0.94 if title else 1.0
    fig.tight_layout(rect=(0, 0.0, 0.86, top_pad))
    return fig, axes_flat


def plot_per_client_block_composition(
    panels: Sequence[Tuple[str, pd.DataFrame]],
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "",
) -> Tuple[Figure, np.ndarray]:
    """Stacked horizontal bars: per-client distribution over stream blocks.

    Args:
        panels: Sequence of ``(panel_title, comp_df)`` pairs where
            ``comp_df`` is the output of
            `federated.per_client_block_composition` for one variant.
            One subplot is rendered per pair.
        figsize: Optional ``(w, h)`` override.
        title: Optional figure-level title.

    Returns:
        ``(fig, axes)`` where ``axes`` is a 1D array of subplot axes.
    """
    n = len(panels)
    if figsize is None:
        figsize = (5.5 * n + 1.5, 3.6)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes_flat = axes[0]
    for ax, (panel_title, comp) in zip(axes_flat, panels):
        if comp is None or comp.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(panel_title, fontsize=12, loc="left")
            continue
        clients = sorted(comp["client"].unique())
        # Use the cell's first row for client labels (they're constant
        # per client).
        client_labels = [comp[comp["client"] == c]["client_label"].iloc[0]
                         for c in clients]
        # For each client, accumulate stacks in CLIENT_GROUP_ORDER so
        # related blocks sit next to each other in the bar.
        y = np.arange(len(clients))
        offset = np.zeros(len(clients))
        for group in fa.CLIENT_GROUP_ORDER:
            blocks_in_group = (comp[comp["group"] == group]["block"]
                               .drop_duplicates().tolist())
            if not blocks_in_group:
                continue
            color = GROUP_COLORS.get(group, "#888888")
            # Render each block with the same group color but a
            # gradient of edge alpha, so the eye still sees individual
            # blocks within a group but the macro story is the group.
            for i, block in enumerate(sorted(blocks_in_group)):
                widths = []
                for c in clients:
                    sub = comp[(comp["client"] == c) & (comp["block"] == block)]
                    widths.append(float(sub["fraction"].iloc[0]) if not sub.empty else 0.0)
                widths_arr = np.array(widths)
                if widths_arr.sum() == 0:
                    continue
                ax.barh(y, widths_arr, left=offset, color=color,
                        edgecolor="white", linewidth=0.6, alpha=0.85)
                # Annotate the block name in the middle of the segment
                # for the largest bar (so we do not flood the panel).
                idx_largest = int(np.argmax(widths_arr))
                w = widths_arr[idx_largest]
                if w >= 0.10:
                    ax.text(offset[idx_largest] + w / 2, y[idx_largest],
                            block, ha="center", va="center",
                            fontsize=8, color="white",
                            fontweight="bold")
                offset += widths_arr
        ax.set_yticks(y)
        ax.set_yticklabels(client_labels, fontsize=11)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Fraction of client's data", fontsize=11)
        ax.tick_params(axis="x", labelsize=10)
        ax.invert_yaxis()  # C0 on top
        ax.set_title(panel_title, fontsize=13, loc="left")
        ax.grid(True, axis="x", alpha=0.3)
    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
               for g in fa.CLIENT_GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles), fontsize=11, framealpha=0.9)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96 if title else 1.0))
    return fig, axes_flat


# Pastel versions of `CLIENT_COLORS` (blue/green/orange/red).  Light
# enough for dark text to read on top, ordered to keep the per-client
# hue consistent across every federated figure that paints clients.
_CLIENT_RIBBON_COLORS: Tuple[str, ...] = (
    "#c7d8f0",  # pale blue   -- C0 / Q1  (matches CLIENT_COLORS[0])
    "#cfe6cf",  # pale green  -- C1 / Q2  (matches CLIENT_COLORS[1])
    "#ffe0c4",  # pale orange -- C2 / Q3  (matches CLIENT_COLORS[2])
    "#f1c8c8",  # pale red    -- C3 / Q4  (matches CLIENT_COLORS[3])
)


def plot_stream_composition_with_partitions(
    composition: Mapping[str, pd.DataFrame],
    *,
    block_boundaries: Sequence[int] = (),
    client_boundaries: Sequence[Tuple[int, str]] = (),
    composition_palettes: Optional[Mapping[str, Mapping[str, str]]] = None,
    composition_titles: Optional[Mapping[str, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, np.ndarray]:
    """Stream-composition panels with a client-partition header strip.

    Replaces the aggregate per-client bar chart with a time-resolved
    stacked-area view: each panel shows the rolling-window composition
    of the stream (time-of-day, road type, weather) as it evolves from
    left to right.  A short header axis above the data panels carries
    one colored ribbon per client with the client label inside, so the
    partition is visible without overlapping the composition data or
    competing with the per-panel legends.  Thin grey dashed lines mark
    individual stream-block transitions; slim orange tick markers along
    the top edge of the top composition panel mark the partition
    boundaries for visual continuity with the ribbon strip.

    Args:
        composition: ``{field: wide_df}`` from
            `streaming.stream_composition`.  Rows are ``items_start``
            (window left edge); columns are category names holding
            fractions in [0, 1].
        block_boundaries: Stream-block boundary positions in
            post-bootstrap frame coordinates, from
            `streaming.block_boundaries_and_midpoints`.  Used to draw
            thin grey dashed inter-block lines.
        client_boundaries: ``(start_frame_idx, label)`` per client,
            from `federated.curated_client_boundaries` or
            `federated.temporal_client_boundaries`.  Draws the colored
            client ribbons in the header strip.
        composition_palettes: ``{field: {category: color}}`` palette
            overrides.
        composition_titles: ``{field: title_string}`` panel title
            overrides.
        figsize: Optional ``(width, height)`` override.

    Returns:
        ``(fig, axes)`` where ``axes`` is a 1-D array containing the
        client-ribbon axis followed by one axis per composition field.
    """
    items = [(f, df) for f, df in composition.items()
             if df is not None and not df.empty]
    n = len(items)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize or (12.5, 3.0))
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return fig, np.array([ax])

    has_ribbon = bool(client_boundaries)
    if figsize is None:
        figsize = (12.5, 2.0 * n + (0.55 if has_ribbon else 0.0) + 0.8)

    n_rows = n + (1 if has_ribbon else 0)
    height_ratios = ([0.22] if has_ribbon else []) + [1.0] * n
    fig, axes = plt.subplots(
        n_rows, 1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": 0.55, "height_ratios": height_ratios},
        squeeze=False,
    )
    axes_flat = axes[:, 0]
    comp_axes = axes_flat[1:] if has_ribbon else axes_flat
    ribbon_ax = axes_flat[0] if has_ribbon else None

    # X-range: prefer the manifest's block-coordinate span (covers the
    # full post-bootstrap stream); fall back to the data extent.
    if block_boundaries:
        x_lo, x_hi = block_boundaries[0], block_boundaries[-1]
    else:
        x_lo = int(items[0][1].index[0])
        x_hi = int(items[-1][1].index[-1]) + 1
        if client_boundaries:
            x_lo = min(x_lo, client_boundaries[0][0])

    # --- Client-ribbon header ----------------------------------------------
    if ribbon_ax is not None:
        for i, (x_start, label) in enumerate(client_boundaries):
            x_end = (client_boundaries[i + 1][0]
                     if i + 1 < len(client_boundaries) else x_hi)
            color = _CLIENT_RIBBON_COLORS[i % len(_CLIENT_RIBBON_COLORS)]
            ribbon_ax.axvspan(x_start, x_end, ymin=0.05, ymax=0.95,
                              color=color, alpha=0.95, lw=0, zorder=1)
            ribbon_ax.text((x_start + x_end) / 2, 0.5, label,
                           ha="center", va="center", fontsize=12,
                           color="#2b2b2b", fontweight="bold",
                           zorder=3, transform=ribbon_ax.get_xaxis_transform())
        # Slim divider lines between client ribbons.
        for x_start, _ in client_boundaries[1:]:
            ribbon_ax.axvline(x_start, color="white", lw=1.8, zorder=2)
        ribbon_ax.set_yticks([])
        # Keep the shared x-locator alive so the bottom composition
        # panel still renders tick labels; only hide the ribbon row's
        # own tick marks and labels.
        ribbon_ax.tick_params(axis="x", which="both",
                              bottom=False, top=False, labelbottom=False)
        for spine in ribbon_ax.spines.values():
            spine.set_visible(False)
        ribbon_ax.set_ylim(0, 1)

    # --- Composition panels -------------------------------------------------
    for ax, (field, frac_df) in zip(comp_axes, items):
        xs = frac_df.index.to_numpy()
        palette = (composition_palettes or {}).get(field, {})
        base = np.zeros_like(xs, dtype=float)
        for col in frac_df.columns:
            vals = frac_df[col].to_numpy()
            color = palette.get(col)
            kw: Dict[str, Any] = dict(alpha=0.75, label=col, linewidth=0)
            if color is not None:
                kw["color"] = color
            ax.fill_between(xs, base, base + vals, **kw)
            base = base + vals

        if block_boundaries:
            for x in list(block_boundaries)[1:-1]:
                ax.axvline(x, color="#444444", alpha=0.30, lw=0.5, ls="--",
                           zorder=1)

        # Slim orange tick markers along the top edge mirror the ribbon
        # boundaries without overlapping the legend or fill.
        if client_boundaries:
            for x_start, _ in client_boundaries[1:]:
                ax.axvline(x_start, color="#e05c00", lw=1.2, alpha=0.85,
                           ymin=0.92, ymax=1.0, zorder=4)

        ax.set_ylabel("Fraction", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_ylim(0, 1)
        ttl = (composition_titles or {}).get(field, f"{field} composition")
        ax.set_title(ttl, fontsize=13, loc="left", pad=6)
        ax.legend(fontsize=11, loc="upper left", framealpha=0.92,
                  ncol=min(4, max(1, len(frac_df.columns))))
        ax.grid(False)

    comp_axes[-1].set_xlabel("Frame index (post-bootstrap)", fontsize=13)
    comp_axes[-1].set_xlim(x_lo, x_hi)
    fig.tight_layout()
    return fig, axes_flat


# =============================================================================
# Per-block accept rate (federated analogue of streaming fig 01)
# =============================================================================

def plot_per_block_accept_rate(
    rates: pd.DataFrame,
    *,
    filter_variants: Optional[Sequence[str]] = None,
    static_variant: Optional[str] = None,
    block_order: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (9.0, 5.0),
    show_std: bool = True,
    ymax: float = 0.45,
    title: str = "",
) -> Tuple[Figure, Axes]:
    """Per-block mean accept rate, one line per filter variant.

    Federated analogue of the streaming "per-block routing" figure
    (`figures.streaming.plot_per_block_routing_lines`).  Filters are
    drawn as colored polylines along the block axis on the left
    y-axis; the optional static variant lives on a twinned right-hand
    axis (typically ~0.6-1.0) so it does not compress the filter
    polylines.  Random is intentionally not drawn -- random's accept
    rate is uniform across blocks by construction.

    Args:
        rates: Long-format DataFrame from
            `federated.per_block_accept_rate_table`.
        filter_variants: Variants to draw as polylines on the left
            axis.  Defaults to all non-random, non-static variants in
            ``rates``.
        static_variant: When provided, draws the static-filter trace
            on a twinned right-hand axis so the filter polylines on
            the left axis stay zoomed in.
        block_order: Optional explicit block order along the x-axis;
            defaults to the sorted unique blocks in ``rates``.
        figsize: ``(w, h)`` for the panel.
        show_std: If True and ``accept_rate_std`` is present, draws a
            seed-std band around each line.
        ymax: Upper limit for the left (filter) y-axis.
        title: Optional panel title.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if rates is None or rates.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return fig, ax
    if block_order is None:
        block_order = sorted(rates["block"].unique())
    block_order = list(block_order)
    x = np.arange(len(block_order))

    rates_by_v = {v: g for v, g in rates.groupby("variant", sort=False)}
    if filter_variants is None:
        filter_variants = [v for v in rates_by_v
                           if v != static_variant
                           and "_random_" not in v]
    handles, labels = [], []
    for v in filter_variants:
        grp = rates_by_v.get(v)
        if grp is None:
            continue
        sub = grp.set_index("block").reindex(block_order)
        y = sub["accept_rate"].astype(float).values
        c = variant_color(v)
        ls = variant_linestyle(v)
        line, = ax.plot(x, y, color=c, linestyle=ls, marker="o",
                        markersize=5, markeredgecolor="white",
                        markeredgewidth=0.4, linewidth=1.5, alpha=0.95)
        handles.append(line)
        labels.append(fa.label_for(v))
        if show_std and "accept_rate_std" in sub.columns:
            std = sub["accept_rate_std"].fillna(0).astype(float).values
            ax.fill_between(x, y - std, y + std, color=c, alpha=0.13,
                            linewidth=0)

    if static_variant and static_variant in rates_by_v:
        ax2 = ax.twinx()
        grp = rates_by_v[static_variant]
        sub = grp.set_index("block").reindex(block_order)
        y = sub["accept_rate"].astype(float).values
        c = variant_color(static_variant)
        line, = ax2.plot(x, y, color=c, linestyle=variant_linestyle(static_variant),
                         marker="D", markersize=5, markeredgecolor="white",
                         markeredgewidth=0.4, linewidth=1.3, alpha=0.85)
        if show_std and "accept_rate_std" in sub.columns:
            std = sub["accept_rate_std"].fillna(0).astype(float).values
            ax2.fill_between(x, y - std, y + std, color=c, alpha=0.10,
                             linewidth=0)
        ax2.set_ylim(0.5, 1.02)
        ax2.set_ylabel(f"Accept rate ({fa.label_for(static_variant)})",
                       color=c, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=c, labelsize=10)
        ax2.spines["top"].set_visible(False)
        handles.append(line)
        labels.append(fa.label_for(static_variant))

    ax.set_xticks(x)
    ax.set_xticklabels(block_order, rotation=40, ha="right", fontsize=11,
                       color="#222222")
    ax.set_ylabel("Accept rate", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, ymax)
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=13, loc="left")
    if handles:
        fig.legend(handles, labels, fontsize=11, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=min(len(handles), 4),
                   framealpha=0.92, columnspacing=1.5, handlelength=1.8,
                   frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig, ax


# =============================================================================
# Per-round per-client accept-rate dynamics (federated analogue of
# streaming accept-rate-over-time top panel)
# =============================================================================

def plot_per_round_per_client_accept(
    panels: Sequence[Tuple[str, pd.DataFrame]],
    *,
    figsize: Optional[Tuple[float, float]] = None,
    smoothing_window: int = 1,
    show_std: bool = True,
    ymax: Optional[float] = None,
    title: str = "",
    n_cols: Optional[int] = None,
) -> Tuple[Figure, np.ndarray]:
    """Per-round per-client accept-rate trajectories, faceted by variant.

    Args:
        panels: Sequence of ``(panel_title, per_round_df)`` pairs;
            ``per_round_df`` is the subset of
            `federated.per_round_per_client_accept` for one variant.
            One subplot is rendered per pair.  Pass an empty
            ``panel_title`` to suppress per-panel titles (sensible for
            single-panel figures whose context comes from the caption).
        figsize: Optional ``(w, h)`` override.
        smoothing_window: Centered rolling-mean window applied to each
            client's trace.  Set to ``1`` to disable.
        show_std: If True and ``accept_rate_std`` is present, draws a
            faint seed-std band per client.
        ymax: Optional shared y-axis upper limit.
        title: Optional figure-level title.
        n_cols: Number of subplot columns.  Defaults to ``len(panels)``
            (single-row layout); set to e.g. ``2`` to wrap into a
            multi-row grid (useful for 2 x 2 multi-filter views).
    """
    n = len(panels)
    cols = n_cols or n
    rows = (n + cols - 1) // cols
    if figsize is None:
        figsize = (5.0 * cols + 1.0, 3.6 * rows + 0.6)
    # Share x (rounds align across filters) but not y: filters with
    # very different equilibria (e.g. static saturating near 1.0 vs
    # reservoir damping toward 0.0) would otherwise compress each
    # other into illegible bands.
    fig, axes = plt.subplots(rows, cols, figsize=figsize,
                             squeeze=False, sharey=False, sharex=True)
    axes_flat = axes.flatten()
    for ax, (panel_title, sub) in zip(axes_flat, panels):
        if sub is None or sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            if panel_title:
                ax.set_title(panel_title, fontsize=12, loc="left")
            continue
        for cid, grp in sub.groupby("client"):
            grp = grp.sort_values("round")
            y = grp["accept_rate"].astype(float)
            if smoothing_window and smoothing_window > 1:
                y = y.rolling(smoothing_window, center=True,
                              min_periods=1).mean()
            color = CLIENT_COLORS.get(int(cid), "#888888")
            label = grp["client_label"].iloc[0]
            ax.plot(grp["round"], y, color=color, linewidth=1.8,
                    alpha=0.92, label=label)
            if show_std and "accept_rate_std" in grp.columns:
                std = grp["accept_rate_std"].fillna(0).astype(float)
                if smoothing_window and smoothing_window > 1:
                    std = std.rolling(smoothing_window, center=True,
                                      min_periods=1).mean()
                ax.fill_between(grp["round"], y - std, y + std,
                                color=color, alpha=0.13, linewidth=0)
        if panel_title:
            ax.set_title(panel_title, fontsize=13, loc="left")
        ax.tick_params(labelsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        if ymax is not None:
            ax.set_ylim(0, ymax)
    # Hide unused axes when n < rows*cols.
    for k in range(n, len(axes_flat)):
        axes_flat[k].axis("off")
    # Bottom-row x-labels and first-column y-labels.
    for k, ax in enumerate(axes_flat[:n]):
        row = k // cols
        col = k % cols
        if row == rows - 1 or k + cols >= n:
            ax.set_xlabel("Communication round", fontsize=12)
        if col == 0:
            ax.set_ylabel("Per-client accept rate", fontsize=12)
    # Shared bottom legend; pull labels from the first non-empty axis.
    # Use the union of all clients across panels so it stays correct
    # for partition-mixed layouts (curated + temporal in one figure).
    seen, handles, labels = set(), [], []
    for ax in axes_flat[:n]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in seen:
                seen.add(ll)
                handles.append(hh)
                labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
                   fontsize=11, framealpha=0.9, frameon=False)
    if title:
        fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    bottom = 0.10 if handles else 0.04
    fig.tight_layout(rect=(0, bottom, 1, 0.95 if title else 1.0))
    return fig, axes_flat


# =============================================================================
# Refresh dynamics (within-stream accept-rate decay diagnostic)
# =============================================================================

def plot_refresh_segment_accept(
    segments: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (8.0, 4.0),
    title: str = "",
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
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.85, ncol=2)
    fig.tight_layout()
    return fig, ax
