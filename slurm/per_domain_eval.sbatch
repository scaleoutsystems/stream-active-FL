#!/bin/bash -l
#SBATCH -A p201222
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00
#SBATCH --job-name=per_domain_eval
#SBATCH --output=/project/home/p201222/data/ZOD_frames_preprocessed/logs/per_domain_eval_%j.out
#SBATCH --error=/project/home/p201222/data/ZOD_frames_preprocessed/logs/per_domain_eval_%j.err

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

# Usage:
#   sbatch slurm/per_domain_eval.sbatch
# Optional env overrides:
#   RUNS_ROOT      (default outputs/streaming)
#   MANIFEST       (default manifest_cityday_curated_boot2000.json under Frames_1600x480)
#   CHECKPOINTS    (default "best final")
#   VARIANT_FILTER (substring match; used when VARIANTS is unset)
#   VARIANTS       (space-separated exact variant dir names; takes precedence)
#   EXTRA_ARGS     (additional flags, e.g. "--force" or "--seed-pattern seed_42")
RUNS_ROOT="${RUNS_ROOT:-outputs/streaming}"
MANIFEST="${MANIFEST:-data/ZOD_frames_preprocessed/Frames_1600x480/manifest_cityday_curated_boot2000.json}"
CHECKPOINTS="${CHECKPOINTS:-best final}"
VARIANT_FILTER="${VARIANT_FILTER:-cityday_curated}"
EXTRA_ARGS="${EXTRA_ARGS:---skip-existing}"

CMD=(python -u tools/per_domain_eval.py
     --runs-root "$RUNS_ROOT"
     --manifest "$MANIFEST"
     --checkpoints $CHECKPOINTS)
if [[ -n "${VARIANTS:-}" ]]; then
  CMD+=(--variants $VARIANTS)
else
  CMD+=(--variant-pattern "$VARIANT_FILTER")
fi
CMD+=($EXTRA_ARGS)
"${CMD[@]}"
