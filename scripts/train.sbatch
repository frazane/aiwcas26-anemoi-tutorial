#!/bin/bash
#SBATCH --job-name=anemoi-train
#SBATCH --account=tra26_aiwcas
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one task per GPU (Lightning maps tasks -> GPUs)
#SBATCH --gres=gpu:1                 # = system.hardware.num_gpus_per_node
#SBATCH --cpus-per-task=8            # dataloader workers live here; billing is per core
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out           # relative to the submit dir; scripts/submit.sh overrides it
#
# Generic 1-GPU anemoi-training job (AIWCAS 2026 tutorial). Students: use `aiwcas train`, which
# validates the config and calls scripts/submit.sh, which puts the log in $SCRATCH/aiwcas26/logs:
#
#   aiwcas train exercises/A_processor_zoo/my_run.yaml [--name RUN] [hydra overrides...]
#   scripts/submit.sh exercises/A_processor_zoo/my_run.yaml [--name RUN] [hydra overrides...]
#
# Direct use also works (log then lands in the directory you submit from):
#
#   sbatch scripts/train.sbatch exercises/A_processor_zoo/solution.yaml training.max_steps=200
#
# Arguments: <config.yaml> [hydra overrides...]. Relative config paths are resolved against the
# submit directory, because `anemoi-training train --config-path` must be absolute.
# Environment: ANEMOI_VENV (default: the shared env in $FAST/aiwcas26/envs/anemoi),
#              ANEMOI_ENV_MODE=auto|sqfs|venv (default auto: squashfs image if there is one, see
#              scripts/anemoi_env.sh), REPO_DIR (default: the submit dir),
#              SCOREBOARD=0 to skip the scoreboard row, SCOREBOARD_DIR to use another board.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch scripts/train.sbatch <config.yaml> [hydra overrides...]" >&2
    exit 2
fi
CONFIG="$1"; shift
[[ "$CONFIG" = /* ]] || CONFIG="${SLURM_SUBMIT_DIR:-$PWD}/$CONFIG"
CONFIG="$(realpath "$CONFIG")"
CONFIG_DIR="$(dirname "$CONFIG")"
CONFIG_NAME="$(basename "$CONFIG" .yaml)"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Shared env, by default mounted from its squashfs image (imports from one file, not ~57k on Lustre)
source "$REPO_DIR/scripts/anemoi_env.sh"

LOG_DIR="${AIWCAS_LOGS:-$SCRATCH/aiwcas26/logs}"
mkdir -p "$LOG_DIR"
GPU_CSV="$LOG_DIR/${SLURM_JOB_NAME:-train}-${SLURM_JOB_ID:-0}.gpu.csv"
STDOUT="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oP 'StdOut=\K\S+' || true)"

# Offline / no-internet settings for compute nodes
export MLFLOW_ALLOW_FILE_STORE=true    # MLflow >= 3.x refuses the ./mlruns file store otherwise
export HYDRA_FULL_ERROR=1              # full tracebacks for config errors
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/.cache/triton}"   # keep JIT caches out of $HOME

echo "== $(date -Is) job ${SLURM_JOB_ID} on $(hostname)"
echo "== config: $CONFIG"
echo "== overrides: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Sample GPU memory every 2 s in the background (peak memory for the scoreboard)
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 2 > "$GPU_CSV" &
SMI_PID=$!

START=$(date +%s)
set +e
# srun gives Lightning the SLURM environment (ranks, devices).
srun anemoi-training train \
    --config-path "$CONFIG_DIR" --config-name "$CONFIG_NAME" \
    diagnostics.print_memory_summary=True \
    "$@"
RC=$?
set -e
END=$(date +%s)
kill "$SMI_PID" 2>/dev/null || true

echo "== training exit code: $RC"
echo "== peak GPU memory used (MiB): $(cut -d, -f3 "$GPU_CSV" | tr -dc '0-9\n' | sort -n | tail -1)"
echo "== training command wall time: $((END - START)) s"

if [[ $RC -eq 0 && "${SCOREBOARD:-1}" != 0 ]]; then
    python "$REPO_DIR/scripts/scoreboard.py" add --job "$SLURM_JOB_ID" \
        --log "$STDOUT" --gpu-csv "$GPU_CSV" --train-start "$START" --train-end "$END" \
        --config "$CONFIG" || echo "== (scoreboard append failed; run it by hand: python scripts/scoreboard.py add --job $SLURM_JOB_ID)"
fi
echo "== $(date -Is) done"
exit $RC
