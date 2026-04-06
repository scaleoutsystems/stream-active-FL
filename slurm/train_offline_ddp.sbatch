#!/bin/bash -l
#SBATCH -A p201222
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH --job-name=offline_ddp
#SBATCH --output=/project/home/p201222/data/ZOD_frames_preprocessed/logs/offline_ddp_%j.out
#SBATCH --error=/project/home/p201222/data/ZOD_frames_preprocessed/logs/offline_ddp_%j.err

# Multi-GPU offline baseline using DistributedDataParallel (4x A100).
# ~3-4x faster than single-GPU. For single-GPU use train_offline.sbatch.

set -euo pipefail

module --force purge
module load env/staging/2023.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

cd ~/stream-active-FL
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export STREAM_ACTIVE_FL_DATA_ROOT=/project/home/p201222/data
export STREAM_ACTIVE_FL_ZOD_ROOT=/project/home/p201222/data/ZOD_clone_2018_scaleout_zenseact
export STREAM_ACTIVE_FL_PREPROCESSED_ROOT=/project/home/p201222/data/ZOD_frames_preprocessed

mkdir -p "$STREAM_ACTIVE_FL_PREPROCESSED_ROOT/logs"
python -c "import sys, torch; print(f'python={sys.executable}  torch={torch.__version__}  cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}')"

export OMP_NUM_THREADS=2

torchrun --nproc_per_node=4 \
  experiments/offline_baseline.py \
  --config configs/offline_baseline.yaml
