# stream-active-FL

<p align="center">
  <img src="assets/cover_art.png" alt="Streaming active learning across road environments" width="720">
</p>

Buffer-based streaming active learning for object detection, with federated learning simulation.
Frames are filtered by a distribution-based novelty scorer before entering the training buffer,
and compared against no-filter and random baselines in both centralized and federated settings.
Experiments use [ZOD Frames](https://zod.zenseact.com).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e '.[notebook]'   # Jupyter / analysis
pip install pytest ruff        # Dev tooling
```

## Data

```bash
# Crop + resize ZOD frames and extract annotations
python tools/preprocessing/prepare_data.py --zod-root /path/to/zod --version full

# Build train/val manifests (base + shared-bootstrap variants)
python tools/preprocessing/build_manifests.py \
    --generate-bootstraps --generate-shared-manifests
```

If the data lives outside the repo, set the following environment variables:

```bash
export STREAM_ACTIVE_FL_DATA_ROOT=/path/to/data
export STREAM_ACTIVE_FL_ZOD_ROOT=/path/to/data/ZOD_clone
export STREAM_ACTIVE_FL_PREPROCESSED_ROOT=/path/to/data/ZOD_frames_preprocessed
```

## Running experiments

```bash
# Streaming detection (bootstrap + online filtering)
python experiments/streaming.py \
    --config configs/streaming/adaptive_reservoir_p20_twoRef_cityday_curated.yaml

# Federated streaming (FedAvg over client-local streams)
python experiments/federated.py \
    --config configs/federated/fed_adaptive_reservoir_p20_twoRef_cityday_curated.yaml

# Offline baseline (performance ceiling)
python experiments/offline.py --config configs/offline/baseline.yaml
```

To reuse an existing bootstrap (e.g. run the temporal variant without retraining):

```bash
python experiments/streaming.py \
    --config configs/streaming/adaptive_reservoir_p20_twoRef_cityday_temporal.yaml \
    --bootstrap-run-dir outputs/streaming/adaptive_reservoir_p20_twoRef_cityday_curated/<run_id>
```

See `configs/streaming/` and `configs/federated/` for all available configurations.

## Output

Each run writes to `outputs/<pipeline>/<variant>/<timestamp>/` and contains a config
snapshot, model checkpoints, per-checkpoint CSVs, and run metadata.

## Analysis

Results are analyzed through two generated notebooks backed by a reusable analysis package:

```bash
# Build (without executing)
python tools/build_streaming_notebook.py
python tools/build_federated_notebook.py

# Build and execute end-to-end
python tools/build_streaming_notebook.py --execute
python tools/build_federated_notebook.py --execute
```

Figures and summary tables land under `reports/streaming/` and `reports/federated/`.
See [`notebooks/README.md`](notebooks/README.md) for the full workflow.
