#!/bin/bash -l
#SBATCH -A p201222
#SBATCH -p gpu
#SBATCH -q short
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH --job-name=tests
#SBATCH --output=/project/home/p201222/data/ZOD_frames_preprocessed/logs/tests_%j.out
#SBATCH --error=/project/home/p201222/data/ZOD_frames_preprocessed/logs/tests_%j.err

set -euo pipefail

module --force purge
module load env/staging/2023.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

mkdir -p /project/home/p201222/data/ZOD_frames_preprocessed/logs

cd ~/stream-active-FL
source .venv/bin/activate

export STREAM_ACTIVE_FL_DATA_ROOT=/project/home/p201222/data
export STREAM_ACTIVE_FL_ZOD_ROOT=/project/home/p201222/data/ZOD_clone_2018_scaleout_zenseact
export STREAM_ACTIVE_FL_PREPROCESSED_ROOT=/project/home/p201222/data/ZOD_frames_preprocessed

srun python -m pytest tests/ -v
