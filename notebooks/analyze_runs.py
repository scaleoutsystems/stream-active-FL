"""Produce summary tables for streaming / federated runs.

Regenerates the tables that live in the streaming and federated
notebooks (per-seed summary, cross-seed aggregate, per-block accept
rate, inter-refresh accept rate) without opening a notebook.  Useful
for CI, reports, and quick sanity checks after a new batch of runs.

Usage:

    python notebooks/analyze_runs.py \\
        --outputs outputs \\
        --pipeline streaming \\
        [--csv-dir reports/runs] \\
        [--variants no_filter_cityday_curated static_p15_cityday_curated ...]

Without --csv-dir the tables are only printed; with it they are also
written as CSVs under that directory.  With --pipeline federated the
script analyzes the federated outputs; --pipeline both runs the two
pipelines back to back.

All heavy lifting lives in analysis_helpers; this script is a thin
driver so notebooks and CI can share the same code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_helpers as ah


def _round_floats(df: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(ndigits)
    return out


def _write(df: pd.DataFrame, path: Optional[Path], label: str) -> None:
    print(f"\n=== {label} ===")
    if df.empty:
        print("  (empty)")
        return
    print(df.to_string(index=False))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"  -> {path}")


def _min_compute_steps(
    runs_df: pd.DataFrame,
    pipeline: str,
    variants: List[str],
) -> dict[str, int]:
    """Smallest final training-step count per manifest family, across filter runs.

    For streaming runs this reads checkpoints.csv (effective compute =
    items processed); for federated runs it reads rounds.csv (sum of
    per-client optimizer steps).  The resulting budget is the compute cap
    at which the no-filter baseline is evaluated for iso-compute mAP.
    """
    out: dict[str, int] = {}
    for variant in variants:
        for rdir in ah.pick_runs_by_seed(runs_df, pipeline, variant).values():
            cfg = ah.load_run_config(rdir)
            fam = ah.filter_mode(cfg)
            if fam not in {"static", "window", "reservoir"}:
                continue
            df = ah.read_csv(rdir / "checkpoints.csv")
            if df is None or df.empty:
                df = ah.read_csv(rdir / "rounds.csv")
            steps_series = ah.compute_step_series(df)
            if steps_series is None or steps_series.dropna().empty:
                continue
            man = ah.manifest_family(cfg)
            steps = int(steps_series.dropna().iloc[-1])
            out[man] = min(out.get(man, steps), steps)
    return out


def analyze_pipeline(
    outputs_root: Path,
    pipeline: str,
    variants: Optional[List[str]] = None,
    csv_dir: Optional[Path] = None,
) -> None:
    runs_df = ah.discover_runs(outputs_root)
    if runs_df.empty:
        print(f"No runs under {outputs_root}.")
        return
    available = sorted(
        runs_df.loc[runs_df["pipeline"] == pipeline, "variant"].unique()
    )
    if not available:
        print(f"No runs for pipeline={pipeline}.")
        return
    if variants is None:
        variants = available
    else:
        missing = [v for v in variants if v not in available]
        if missing:
            print(f"Skipping missing variants: {missing}")
        variants = [v for v in variants if v in available]
    if not variants:
        print(f"No matching variants for pipeline={pipeline}.")
        return

    base = (csv_dir / pipeline) if csv_dir else None
    steps_by_manifest = _min_compute_steps(runs_df, pipeline, variants)

    # (1) Per-(variant, seed) summary (uses the same helper as the notebooks).
    per_seed = ah.variant_summary_table(
        runs_df, pipeline, variants,
        target_optim_steps=steps_by_manifest,
    )
    _write(_round_floats(per_seed.drop(columns=["run_dir"], errors="ignore")),
           base / "per_seed_summary.csv" if base else None,
           f"{pipeline} per-(variant, seed) summary")
    if not per_seed.empty:
        agg = ah.aggregate_summary_across_seeds(per_seed)
        _write(_round_floats(agg),
               base / "variant_summary_agg.csv" if base else None,
               f"{pipeline} variant summary aggregated across seeds")

    project_root = ah.find_project_root(Path(__file__).resolve().parent)

    # (2) Per-block accept-rate tables (one per variant, latest seed)
    per_block_rows: list[pd.DataFrame] = []
    for variant in variants:
        rdir = ah.pick_latest_run(runs_df, pipeline, variant)
        if rdir is None:
            continue
        cfg = ah.load_run_config(rdir)
        man = ah.load_manifest(project_root, str(cfg.get("manifest_path", "")))
        if man is None:
            continue
        boot_n = ah.get_bootstrap_size(man, cfg)
        enr = ah.load_enriched_streaming_decisions(rdir, project_root)
        tbl = ah.per_block_accept_rate(enr, man, bootstrap_frames=boot_n)
        if tbl.empty:
            continue
        tbl = tbl.copy()
        tbl.insert(0, "variant", variant)
        tbl.insert(1, "filter_family", ah.filter_mode(cfg))
        per_block_rows.append(tbl)
    if per_block_rows:
        per_block = pd.concat(per_block_rows, ignore_index=True)
        _write(_round_floats(per_block),
               base / "per_block_accept_rate.csv" if base else None,
               f"{pipeline} per-block accept rate")

    # (3) Inter-refresh accept rate (window / reservoir runs only)
    seg_rows: list[pd.DataFrame] = []
    for variant in variants:
        rdir = ah.pick_latest_run(runs_df, pipeline, variant)
        if rdir is None:
            continue
        cfg = ah.load_run_config(rdir)
        fam = ah.filter_mode(cfg)
        if fam not in {"window", "reservoir"}:
            continue
        enr = ah.load_enriched_streaming_decisions(rdir, project_root)
        seg = ah.refresh_accept_rate_segments(enr, ah.load_refreshes(rdir))
        if seg.empty:
            continue
        seg = seg.copy()
        seg.insert(0, "variant", variant)
        seg.insert(1, "filter_family", fam)
        seg_rows.append(seg)
    if seg_rows:
        segs = pd.concat(seg_rows, ignore_index=True)
        _write(_round_floats(segs),
               base / "inter_refresh_accept.csv" if base else None,
               f"{pipeline} inter-refresh accept rates")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--outputs", default="outputs",
                    help="Path to outputs root (default: outputs/).")
    ap.add_argument("--pipeline", choices=["streaming", "federated", "both"],
                    default="both")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Optional list of variant names to include. "
                         "Omit to include every discovered variant.")
    ap.add_argument("--csv-dir", default=None,
                    help="If set, write tables as CSV under this directory.")
    args = ap.parse_args()

    outputs = Path(args.outputs).resolve()
    csv_dir = Path(args.csv_dir).resolve() if args.csv_dir else None
    pipelines = ["streaming", "federated"] if args.pipeline == "both" else [args.pipeline]
    for pl in pipelines:
        analyze_pipeline(outputs, pl, variants=args.variants, csv_dir=csv_dir)


if __name__ == "__main__":
    main()
