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
    --config configs/detection/no_filter.yaml
```

## Experiments

### Offline baseline

Multi-epoch shuffled training on the full dataset:

```bash
python experiments/offline_baseline.py \
    --config configs/detection/offline_baseline.yaml
```

