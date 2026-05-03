#!/bin/bash -l
#SBATCH -A p201222
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --job-name=fed_det
#SBATCH --output=/project/home/p201222/data/ZOD_frames_preprocessed/logs/fed_det_%j.out
#SBATCH --error=/project/home/p201222/data/ZOD_frames_preprocessed/logs/fed_det_%j.err

set -euo pipefail

module --force purge
module load env/staging/2023.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

cd ~/stream-active-FL
source .venv/bin/activate

export STREAM_ACTIVE_FL_DATA_ROOT=/project/home/p201222/data
export STREAM_ACTIVE_FL_ZOD_ROOT=/project/home/p201222/data/ZOD_clone_2018_scaleout_zenseact
export STREAM_ACTIVE_FL_PREPROCESSED_ROOT=/project/home/p201222/data/ZOD_frames_preprocessed

mkdir -p "$STREAM_ACTIVE_FL_PREPROCESSED_ROOT/logs"
python -c "import sys, torch; print(f'python={sys.executable}  torch={torch.__version__}  cuda={torch.cuda.is_available()}')"

# Pass CONFIG_FILE at submit time to select experiment:
#   sbatch --export=ALL,CONFIG_FILE=configs/federated/fed_dist_thresh_cityday_road_type_p10.yaml slurm/train_federated.sbatch
# Optional SEED override (nests run under seed_<N>/<timestamp>/):
#   sbatch --export=ALL,CONFIG_FILE=...,SEED=43 slurm/train_federated.sbatch
# Optional BOOTSTRAP_RUN_DIR to reuse a bootstrap:
#   sbatch --export=ALL,CONFIG_FILE=...,BOOTSTRAP_RUN_DIR=outputs/federated/fed_no_filter_cityday_road_type/<run_id> slurm/train_federated.sbatch
CONFIG_FILE="${CONFIG_FILE:-configs/federated/fed_no_filter_cityday_road_type.yaml}"
echo "Config: $CONFIG_FILE"

CMD=(python experiments/federated.py --config "$CONFIG_FILE")
if [[ -n "${BOOTSTRAP_RUN_DIR:-}" ]]; then
  # Resolve globs (e.g. "outputs/.../seed_43/*") to the newest matching dir,
  # so dependent jobs can be queued before the producing run exists.
  if [[ "$BOOTSTRAP_RUN_DIR" == *"*"* ]]; then
    CANDIDATE=$(ls -1dt $BOOTSTRAP_RUN_DIR 2>/dev/null | head -n1 || true)
    if [[ -z "$CANDIDATE" ]]; then
      echo "ERROR: no directory matches BOOTSTRAP_RUN_DIR pattern: $BOOTSTRAP_RUN_DIR" >&2
      exit 1
    fi
    BOOTSTRAP_RUN_DIR="$CANDIDATE"
  fi
  echo "Reusing bootstrap: $BOOTSTRAP_RUN_DIR"
  CMD+=(--bootstrap-run-dir "$BOOTSTRAP_RUN_DIR")
fi
if [[ -n "${SEED:-}" ]]; then
  echo "Seed override: $SEED"
  CMD+=(--seed "$SEED")
fi
"${CMD[@]}"
