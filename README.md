# stream-active-FL

Buffer-based streaming active learning for object detection, with federated
learning simulation. Compares active filtering strategies (distribution,
uncertainty, gradient-norm, random) against accept-all baselines in both
centralized streaming and federated settings, using ZOD Frames.

## Setup

```bash
pip install -e .
```

## Data Preprocessing

```bash
# Crop + resize ZOD frames and extract annotations
python tools/preprocessing/prepare_data.py --zod-root /path/to/zod --version full

# Build train/val manifests
python tools/preprocessing/build_manifests.py
```

If your data lives outside the repo, set these environment variables:

```bash
export STREAM_ACTIVE_FL_DATA_ROOT=/path/to/data
export STREAM_ACTIVE_FL_ZOD_ROOT=/path/to/data/ZOD_clone_2018_scaleout_zenseact
export STREAM_ACTIVE_FL_PREPROCESSED_ROOT=/path/to/data/ZOD_frames_preprocessed
```

## Running Experiments

```bash
# Offline baseline (performance ceiling)
python experiments/offline_baseline.py --config configs/offline_baseline.yaml

# Streaming detection (bootstrap + online filtering)
python experiments/streaming_detection.py --config configs/streaming_distribution_filter.yaml

# Federated streaming (FedAvg over client-local streaming)
python experiments/federated_detection.py --config configs/federated_no_filter.yaml
```

Streaming and federated experiments can reuse a previous bootstrap to save time:

```bash
python experiments/streaming_detection.py \
    --config configs/streaming_uncertainty_filter.yaml \
    --bootstrap-run-dir outputs/streaming/no_filter/<run_id>
```

See `configs/` for the full set of experiment configurations.

## Output

Each run writes to `outputs/<pipeline>/<variant>/<timestamp>/` with config
snapshots, model checkpoints, per-epoch/checkpoint CSVs, and run metadata.
