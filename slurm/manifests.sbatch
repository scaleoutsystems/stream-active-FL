#!/bin/bash -l
#SBATCH -A p201222
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH --job-name=build_manifests
#SBATCH --output=/project/home/p201222/data/ZOD_frames_preprocessed/logs/build_manifests_%j.out
#SBATCH --error=/project/home/p201222/data/ZOD_frames_preprocessed/logs/build_manifests_%j.err

set -euo pipefail

module --force purge
module load env/staging/2023.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Avoid module-provided Python path precedence issues.
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

cd ~/stream-active-FL
source .venv/bin/activate

export STREAM_ACTIVE_FL_DATA_ROOT=/project/home/p201222/data
export STREAM_ACTIVE_FL_ZOD_ROOT=/project/home/p201222/data/ZOD_clone_2018_scaleout_zenseact
export STREAM_ACTIVE_FL_PREPROCESSED_ROOT=/project/home/p201222/data/ZOD_frames_preprocessed

LOG_DIR="$STREAM_ACTIVE_FL_PREPROCESSED_ROOT/logs"
DATA_DIR="$STREAM_ACTIVE_FL_PREPROCESSED_ROOT/Frames_1600x480"
mkdir -p "$LOG_DIR"

# Default: full regeneration at legacy bootstrap size (5000).
# To regenerate bootstraps + shared manifests at an alternative size,
# override VARIANTS and BOOTSTRAP_N, e.g.:
#   sbatch --export=ALL,VARIANTS="",BOOTSTRAP_N=2000 slurm/manifests.sbatch
# (VARIANTS="" skips the ordering-only manifests, which do not depend on
#  bootstrap size and are unchanged.)
VARIANTS="${VARIANTS:-base temporal road_type road_type_reverse road_type_time conditions urban_rural cityday_curated}"
BOOTSTRAP_N="${BOOTSTRAP_N:-5000}"

CMD=(python tools/preprocessing/build_manifests.py
     --data-dir "$DATA_DIR"
     --zod-root "$STREAM_ACTIVE_FL_ZOD_ROOT"
     --zod-version full
     --bootstrap-n "$BOOTSTRAP_N"
     --generate-bootstraps
     --generate-shared-manifests
     --variants $VARIANTS)
srun "${CMD[@]}"
