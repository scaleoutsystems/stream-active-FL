# stream-active-FL

Buffer-based streaming active learning for object detection, with federated
learning simulation. Compares distribution-based filtering against no-filter
and random baselines in both centralized streaming and federated settings, using ZOD Frames.

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

## Data Preprocessing

```bash
# Crop + resize ZOD frames and extract annotations
python tools/preprocessing/prepare_data.py --zod-root /path/to/zod --version full

# Build train/val manifests (base + shared-bootstrap variants)
python tools/preprocessing/build_manifests.py \
    --generate-bootstraps --generate-shared-manifests
```

If your data lives outside the repo, set these environment variables:

```bash
export STREAM_ACTIVE_FL_DATA_ROOT=/path/to/data
export STREAM_ACTIVE_FL_ZOD_ROOT=/path/to/data/ZOD_clone
export STREAM_ACTIVE_FL_PREPROCESSED_ROOT=/path/to/data/ZOD_frames_preprocessed
```

## Running Experiments

```bash
# Streaming detection (bootstrap + online filtering)
python experiments/streaming.py \
    --config configs/streaming/dist_thresh_cityday_road_type_p10.yaml

# Federated streaming (FedAvg over client-local streams)
python experiments/federated.py \
    --config configs/federated/fed_dist_thresh_cityday_road_type_p10.yaml

# Offline baseline (performance ceiling)
python experiments/offline.py --config configs/offline/baseline.yaml
```

Reuse a previous bootstrap to save time:

```bash
python experiments/streaming.py \
    --config configs/streaming/dist_thresh_cityday_reverse_p10.yaml \
    --bootstrap-run-dir outputs/streaming/dist_thresh_cityday_road_type_p10/<run_id>
```

See `configs/streaming/` and `configs/federated/` for all experiment
configurations.

## Output

Each run writes to `outputs/<pipeline>/<variant>/<timestamp>/` with config
snapshots, model checkpoints, per-checkpoint CSVs, and run metadata.
