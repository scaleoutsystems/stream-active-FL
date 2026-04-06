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
#   sbatch --export=ALL,CONFIG_FILE=configs/federated_distribution_filter.yaml slurm/train_federated.sbatch
# Optional: also pass BOOTSTRAP_RUN_DIR to reuse a bootstrap:
#   sbatch --export=ALL,CONFIG_FILE=configs/federated_distribution_filter.yaml,BOOTSTRAP_RUN_DIR=outputs/federated/no_filter/<run_id> slurm/train_federated.sbatch
CONFIG_FILE="${CONFIG_FILE:-configs/federated_no_filter.yaml}"
echo "Config: $CONFIG_FILE"

if [[ -n "${BOOTSTRAP_RUN_DIR:-}" ]]; then
  echo "Reusing bootstrap: $BOOTSTRAP_RUN_DIR"
  python experiments/federated_detection.py \
    --config "$CONFIG_FILE" \
    --bootstrap-run-dir "$BOOTSTRAP_RUN_DIR"
else
  python experiments/federated_detection.py \
    --config "$CONFIG_FILE"
fi
