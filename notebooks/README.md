# Notebooks

`01_streaming_analysis.ipynb` and `02_federated_analysis.ipynb` are generated from
build scripts — do not edit cell content directly.  Edit the build script or the
underlying analysis package and re-run.

Generated figures and tables land under `reports/streaming/` and `reports/federated/`
(figures as PDF + PNG, tables as CSV).

## Analysis package

The notebooks are thin wrappers around `src/stream_active_fl/analysis/`:

```
analysis/
    runs.py          Run discovery, CSV loading, summaries
    streaming.py     Streaming tables (inventory, iso-accept, ablations)
    federated.py     Federated tables (per-client routing, novelty, ...)
    figures/
        streaming.py Streaming plotters
        federated.py Federated plotters
```

The package imports lazily, so analysis environments do not need PyTorch.

## Build commands

```bash
# Rebuild notebook structure after editing analysis code
python tools/build_streaming_notebook.py
python tools/build_federated_notebook.py

# Rebuild and execute end-to-end
python tools/build_streaming_notebook.py --execute
python tools/build_federated_notebook.py --execute

# Refresh summary tables only (no notebook needed)
python -m stream_active_fl.analysis --summary --csv-dir reports
python -m stream_active_fl.analysis --pipeline federated --summary --csv-dir reports
```

## Adding a new variant

1. Let runs land under `outputs/<pipeline>/<variant>/seed_<N>/<timestamp>/`.
2. Add the variant name to `FEATURED_VARIANTS` and a label to `VARIANT_LABEL` in
   `stream_active_fl.analysis.streaming` or `.federated`.
3. Add entries to `ISO_ACCEPT_PAIRINGS` or `ABLATION_PAIRINGS` if needed.
4. Pick a color in the matching figure module's `VARIANT_COLOR_PALETTE`.
5. Re-run the build script — tables, figures, and notebook cells update automatically.
