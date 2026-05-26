"""
Shared infrastructure for streaming and federated figure modules.

The pipeline-specific submodules (`streaming`, `federated`) provide the
opinionated plotters; this package-level module hosts only the bits both
share so we do not duplicate them.

Public helpers:
- save_figure: write a Matplotlib figure as PDF + PNG into a directory.
- heatmap: render a numeric DataFrame as an annotated heatmap.

Both submodules also share the `FAMILY_COLORS` palette via
`stream_active_fl.analysis.runs.FILTER_FAMILY_COLORS`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    # Matplotlib is an optional runtime dependency; the annotations are
    # only consumed by type-checkers (we use `from __future__ import
    # annotations` so they stay as strings at runtime).
    Axes = Any
    Figure = Any


__all__ = ["save_figure", "heatmap"]


def save_figure(
    fig: Figure,
    name: str,
    *,
    out_dir: Path,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 200,
    bbox_inches: str = "tight",
) -> List[Path]:
    """Save `fig` under `out_dir` as every format in `formats`.

    Args:
        fig: The Matplotlib figure to save.
        name: File stem (no extension); one file per format is written.
        out_dir: Output directory; created lazily if it does not exist.
        formats: Image formats to emit (defaults to PDF + PNG).
        dpi: Resolution for raster formats.
        bbox_inches: Forwarded to `Figure.savefig`.

    Returns:
        The list of paths written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for fmt in formats:
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p, format=fmt, dpi=dpi, bbox_inches=bbox_inches)
        written.append(p)
    return written


def heatmap(
    df: pd.DataFrame,
    *,
    ax: Axes,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    annotate: bool = True,
    fmt: str = "{:.3f}",
    xtick_fontsize: float = 9.5,
    ytick_fontsize: float = 10.0,
    cell_fontsize: float = 9.0,
    title_fontsize: float = 12.0,
    xtick_rotation: float = 30.0,
) -> Any:
    """Render `df` as an annotated heatmap on `ax`.

    Cell text color is auto-picked from background luminance so the
    annotations stay readable across the colormap.

    Args:
        df: Wide-format numeric DataFrame; rows become y-ticks and
            columns become x-ticks.
        ax: The Matplotlib axes to draw on.
        cmap: Colormap name passed to `imshow`.
        vmin, vmax: Color-scale endpoints; default to `df`'s extrema.
        title: Optional axes title (rendered left-aligned).
        annotate: Whether to write the numeric value into each cell.
        fmt: Format string for cell annotations.
        xtick_fontsize: Font size for column labels.
        ytick_fontsize: Font size for row labels.
        cell_fontsize: Font size for in-cell numeric annotations.
        title_fontsize: Font size for the axes title.
        xtick_rotation: Rotation angle (degrees) for column labels.

    Returns:
        The `AxesImage` returned by `imshow`, useful for `colorbar`.
    """
    arr = df.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=xtick_rotation, ha="right",
                       fontsize=xtick_fontsize)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index, fontsize=ytick_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize, loc="left")
    if not annotate:
        return im
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = arr[i, j]
            if pd.isna(v):
                continue
            try:
                rgba = im.cmap(im.norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "white" if lum < 0.5 else "black"
            except Exception:
                color = "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=cell_fontsize, color=color)
    return im
