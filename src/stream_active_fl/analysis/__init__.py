"""
Analysis utilities for streaming and federated experiment outputs.

    The submodules here turn the on-disk artifacts produced by
    `experiments/streaming.py` and `experiments/federated.py` into tidy
    DataFrames and figures for the write-up.

Submodules:
- runs: Generic run discovery, CSV / manifest loading, multi-seed
  aggregation, summary tables.  Pipeline-agnostic; both `streaming`
  and `federated` build on top of it.
- streaming: Streaming-specific tables (variant registry, inventory,
  iso-accept pairings, per-domain grids, ablation comparisons).
- federated: Federated-specific tables (per-client routing, novelty
  ratio, heavy-local schedule comparison) plus the same iso-accept and
  per-domain story as the streaming module.
- figures.streaming, figures.federated: Plotters that consume the
  tables in their respective analysis modules.

A typical notebook session is:

        from stream_active_fl.analysis import runs as ar
    from stream_active_fl.analysis import streaming as sa
    from stream_active_fl.analysis.figures import streaming as sf

    project_root = ar.find_project_root()
    inv = sa.inventory_table(project_root=project_root, tail_k=5)
    fig, _ = sf.plot_inventory_scatter(inv)
"""
