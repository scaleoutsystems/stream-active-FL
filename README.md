# stream-active-FL

Experimental code for a master thesis on **client-side filtering** for continuous learning from streaming perception data in a federated setting. For each incoming stream item, the system decides whether to train, store for replay, or skip, using model-driven selection criteria.

## Quick start

From the repo root:

```bash
# Offline classification baseline (upper bound)
python experiments/offline_classification.py --config configs/offline_classification.yaml

# Streaming classification with replay
python experiments/streaming_classification.py --config configs/streaming_classification_no_filter.yaml
```

Detection experiments use `experiments/streaming_detection.py` and `configs/streaming_detection_*.yaml`.

## Install

Dependencies are in `pyproject.toml`. From the repo root:

```bash
pip install -e .
```

## Status

Early experimental. Codebase and configs are under active development.

## License

See [LICENSE](LICENSE).
