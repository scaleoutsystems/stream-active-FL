# stream-active-FL

Buffer-based streaming active learning for object detection on ZOD Frames.

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Preprocess ZOD Frames (crop + resize + extract annotations)
python tools/preprocessing/prepare_data.py \
    --zod-root /path/to/zod \
    --version full

# 3. Run an experiment
python experiments/streaming_detection.py \
    --config configs/streaming_no_filter.yaml
```

## Experiments

### Offline baseline

Multi-epoch shuffled training on the full dataset:

```bash
python experiments/offline_baseline.py \
    --config configs/offline_baseline.yaml
```

### Streaming (single-machine)

Two-phase streaming experiment (bootstrap + online filtering):

```bash
python experiments/streaming_detection.py \
    --config configs/streaming_distribution_filter.yaml
```

### Federated streaming (simulated FL)

Server-side FedAvg over client-local streaming updates:

```bash
python experiments/federated_detection.py \
    --config configs/federated_no_filter.yaml
```

You can reuse a shared bootstrap from any previous streaming run:

```bash
python experiments/federated_detection.py \
    --config configs/federated_no_filter.yaml \
    --bootstrap-run-dir outputs/streaming/no_filter/<run_id>
```
