#!/bin/bash -l
#SBATCH -A p201222
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH --job-name=prep_zod
#SBATCH --output=/project/home/p201222/data/ZOD_frames_preprocessed/logs/prep_zod_%j.out
#SBATCH --error=/project/home/p201222/data/ZOD_frames_preprocessed/logs/prep_zod_%j.err

set -euo pipefail

module --force purge
module load env/staging/2023.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Avoid module-provided Python path precedence issues.
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

mkdir -p /project/home/p201222/data/ZOD_frames_preprocessed/logs

cd ~/stream-active-FL
source .venv/bin/activate
export STREAM_ACTIVE_FL_DATA_ROOT=/project/home/p201222/data
export STREAM_ACTIVE_FL_ZOD_ROOT=/project/home/p201222/data/ZOD_clone_2018_scaleout_zenseact
export STREAM_ACTIVE_FL_PREPROCESSED_ROOT=/project/home/p201222/data/ZOD_frames_preprocessed
python -c "import typing_extensions, zod; print('typing_extensions=', typing_extensions.__file__, 'zod=', zod.__version__)"
srun python tools/preprocessing/prepare_data.py --version full
