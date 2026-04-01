#!/bin/bash -l
#SBATCH -A p201222
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --job-name=stream_det
#SBATCH --output=/project/home/p201222/data/ZOD_frames_preprocessed/logs/stream_det_%j.out
#SBATCH --error=/project/home/p201222/data/ZOD_frames_preprocessed/logs/stream_det_%j.err

set -euo pipefail

module --force purge
module load env/staging/2023.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

cd ~/stream-active-FL
source .venv/bin/activate

export STREAM_ACTIVE_FL_DATA_ROOT=/project/home/p201222/data
export STREAM_ACTIVE_FL_ZOD_ROOT=/project/home/p201222/data/ZOD_clone_2018_scaleout_zenseact
export STREAM_ACTIVE_FL_PREPROCESSED_ROOT=/project/home/p201222/data/ZOD_frames_preprocessed

mkdir -p "$STREAM_ACTIVE_FL_PREPROCESSED_ROOT/logs"

# Optional: pass BOOTSTRAP_RUN_DIR at submit time to skip bootstrap:
#   sbatch --export=ALL,BOOTSTRAP_RUN_DIR=outputs/streaming/no_filter/<run_id> slurm/train_streaming.sbatch
if [[ -n "${BOOTSTRAP_RUN_DIR:-}" ]]; then
  srun python experiments/streaming_detection.py \
    --config configs/streaming_no_filter.yaml \
    --bootstrap-run-dir "$BOOTSTRAP_RUN_DIR"
else
  srun python experiments/streaming_detection.py \
    --config configs/streaming_no_filter.yaml
fi
