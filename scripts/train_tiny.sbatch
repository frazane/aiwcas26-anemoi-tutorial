#!/bin/bash
#SBATCH --job-name=anemoi-tiny
#SBATCH --account=tra26_aiwcas
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one task per GPU (Lightning maps tasks -> GPUs)
#SBATCH --gres=gpu:1                 # = system.hardware.num_gpus_per_node
#SBATCH --cpus-per-task=8            # dataloader workers live here; billing is per core
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --output=/leonardo_scratch/large/usertrain/a08trd25/aiwcas26/logs/%x-%j.out
# NOTE: #SBATCH lines cannot expand $SCRATCH; edit the --output path above
#       (or pass `sbatch --output=...`) if you are not user a08trd25.
#
# Tiny anemoi-training run on 1 A100 (AIWCAS 2026 tutorial).
#   sbatch scripts/train_tiny.sbatch                       # from the repo root
#   sbatch scripts/train_tiny.sbatch model.num_channels=128 training.max_steps=800
# Extra arguments are passed straight to `anemoi-training train` as Hydra overrides.
set -euo pipefail

module load python/3.11.7
source "${ANEMOI_VENV:-$SCRATCH/envs/anemoi}/bin/activate"

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONFIG_DIR="${CONFIG_DIR:-$REPO_DIR/configs}"     # must be an ABSOLUTE path for `train`
CONFIG_NAME="${CONFIG_NAME:-04_tiny_train}"

mkdir -p "$SCRATCH/aiwcas26/logs" "$SCRATCH/aiwcas26/training"

# Offline / no-internet settings for compute nodes
export MLFLOW_ALLOW_FILE_STORE=true    # MLflow >= 3.x refuses the ./mlruns file store otherwise
export HYDRA_FULL_ERROR=1              # full tracebacks for config errors
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "== $(date -Is) job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Sample GPU utilisation / memory every 10 s in the background
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 \
    > "$SCRATCH/aiwcas26/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv" &
SMI_PID=$!

START=$(date +%s)
# srun gives Lightning the SLURM environment (ranks, devices); with more GPUs use
# --ntasks-per-node=N --gres=gpu:N and system.hardware.num_gpus_per_node=N.
srun anemoi-training train \
    --config-path "$CONFIG_DIR" --config-name "$CONFIG_NAME" \
    diagnostics.print_memory_summary=True \
    "$@"
END=$(date +%s)

kill "$SMI_PID" 2>/dev/null || true
echo "== peak GPU memory used (MiB): $(cut -d, -f3 "$SCRATCH/aiwcas26/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv" | tr -dc '0-9\n' | sort -n | tail -1)"
echo "== training command wall time: $((END - START)) s"
echo "== $(date -Is) done"
