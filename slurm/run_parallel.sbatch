#!/bin/bash -l
#SBATCH -A p201222
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH --job-name=parallel_det
#SBATCH --output=/project/home/p201222/data/ZOD_frames_preprocessed/logs/parallel_det_%j.out
#SBATCH --error=/project/home/p201222/data/ZOD_frames_preprocessed/logs/parallel_det_%j.err

# Run up to 4 experiments in parallel, one per GPU.
#
# Arguments are <type> <config> pairs:
#   sbatch slurm/run_parallel.sbatch \
#     streaming configs/streaming/adaptive_cityday_road_type_p10.yaml \
#     streaming configs/streaming/adaptive_citymix_road_type_p10.yaml \
#     streaming configs/streaming/adaptive_citymix_conditions_p10.yaml
#
# Optional SEED override (applied to every experiment):
#   sbatch --export=ALL,SEED=43 slurm/run_parallel.sbatch streaming <cfg1> streaming <cfg2>
#
# Optional BOOTSTRAP_RUN_DIR override (applied to every experiment):
#   sbatch --export=ALL,BOOTSTRAP_RUN_DIR=outputs/streaming/no_filter_cityday_road_type/<run_id> \
#     slurm/run_parallel.sbatch streaming <cfg1> streaming <cfg2>
#
# Supported types: streaming, federated

set -uo pipefail

module --force purge
module load env/staging/2023.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

cd ~/stream-active-FL
source .venv/bin/activate

export STREAM_ACTIVE_FL_DATA_ROOT=/project/home/p201222/data
export STREAM_ACTIVE_FL_ZOD_ROOT=/project/home/p201222/data/ZOD_clone_2018_scaleout_zenseact
export STREAM_ACTIVE_FL_PREPROCESSED_ROOT=/project/home/p201222/data/ZOD_frames_preprocessed

LOG_DIR="$STREAM_ACTIVE_FL_PREPROCESSED_ROOT/logs"
mkdir -p "$LOG_DIR"

python -c "import sys, torch; print(f'python={sys.executable}  torch={torch.__version__}  cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}')"

# Build array of allocated GPU IDs from Slurm (handles remapped indices)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_IDS=(0 1 2 3)
fi

ARGS=("$@")
NUM_ARGS=${#ARGS[@]}

if [[ $NUM_ARGS -eq 0 || $((NUM_ARGS % 2)) -ne 0 ]]; then
  echo "Usage: sbatch slurm/run_parallel.sbatch <type> <config> [<type> <config> ...]"
  echo "  type: streaming | federated"
  echo "  Max 4 experiments (one per GPU)."
  exit 1
fi

NUM_EXPERIMENTS=$((NUM_ARGS / 2))
if [[ $NUM_EXPERIMENTS -gt 4 ]]; then
  echo "ERROR: max 4 experiments (one per GPU), got $NUM_EXPERIMENTS"
  exit 1
fi

echo "============================================================"
echo "Parallel Launcher: $NUM_EXPERIMENTS experiments on $NUM_EXPERIMENTS GPUs"
if [[ -n "${BOOTSTRAP_RUN_DIR:-}" ]]; then
  echo "Bootstrap reuse: $BOOTSTRAP_RUN_DIR"
fi
echo "============================================================"
echo ""

PIDS=()
for i in $(seq 0 $((NUM_EXPERIMENTS - 1))); do
  TYPE="${ARGS[$((i * 2))]}"
  CONFIG="${ARGS[$((i * 2 + 1))]}"

  case "$TYPE" in
    streaming) SCRIPT="experiments/streaming_detection.py" ;;
    federated) SCRIPT="experiments/federated_detection.py" ;;
    *) echo "ERROR: unknown type '$TYPE' (use streaming or federated)"; exit 1 ;;
  esac

  CONFIG_NAME=$(basename "$CONFIG" .yaml)
  LOGFILE="$LOG_DIR/${CONFIG_NAME}_${SLURM_JOB_ID}"

  echo "  GPU $i (device ${GPU_IDS[$i]}): $TYPE -- $CONFIG"
  echo "         logs: ${LOGFILE}.{out,err}"

  EXTRA_ARGS=()
  if [[ -n "${BOOTSTRAP_RUN_DIR:-}" ]]; then
    EXTRA_ARGS+=(--bootstrap-run-dir "$BOOTSTRAP_RUN_DIR")
  fi
  if [[ -n "${SEED:-}" ]]; then
    EXTRA_ARGS+=(--seed "$SEED")
  fi

  CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" OMP_NUM_THREADS=2 \
    python "$SCRIPT" --config "$CONFIG" "${EXTRA_ARGS[@]}" \
    > "${LOGFILE}.out" 2> "${LOGFILE}.err" &

  PIDS+=($!)
done

echo ""
echo "PIDs: ${PIDS[*]}"
echo "Waiting for all experiments to finish..."
echo ""

FAILED=0
for i in $(seq 0 $((NUM_EXPERIMENTS - 1))); do
  CONFIG="${ARGS[$((i * 2 + 1))]}"
  if wait "${PIDS[$i]}"; then
    echo "  GPU $i: DONE  ($CONFIG)"
  else
    echo "  GPU $i: FAILED ($CONFIG) -- check logs"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
if [[ $FAILED -gt 0 ]]; then
  echo "$FAILED/$NUM_EXPERIMENTS experiments FAILED"
  exit 1
fi
echo "All $NUM_EXPERIMENTS experiments completed successfully."
