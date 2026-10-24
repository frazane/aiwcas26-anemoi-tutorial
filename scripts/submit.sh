#!/bin/bash
# Submit one 1-GPU anemoi-training job for any config file (AIWCAS 2026 tutorial).
#
#   scripts/submit.sh <config.yaml> [--name RUN_NAME] [hydra overrides...]
#
# Examples:
#   scripts/submit.sh exercises/A_processor_zoo/solution.yaml
#   scripts/submit.sh exercises/A_processor_zoo/mine.yaml --name A-gt-128ch model.num_channels=128
#
# What it does for you:
#   * refuses configs that still contain ??? (unfinished TODOs): validate first;
#   * resolves the config to an ABSOLUTE path (`train --config-path` breaks with relative paths);
#   * writes the Slurm log to $SCRATCH/aiwcas26/logs/<job-name>-<jobid>.out (#SBATCH lines
#     cannot expand $SCRATCH, so the path is passed here);
#   * runs scripts/train.sbatch, which appends your run to the room scoreboard when it finishes.
# Environment: ANEMOI_VENV (default: the shared env) and ANEMOI_ENV_MODE (auto|sqfs|venv) are passed to the job.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 || "$1" == -h || "$1" == --help ]]; then
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

CONFIG="$1"; shift
if [[ ! -f "$CONFIG" ]]; then
    echo "submit: config file not found: $CONFIG" >&2
    exit 1
fi
CONFIG="$(realpath "$CONFIG")"
case "$CONFIG" in *.yaml) ;; *) echo "submit: config must be a .yaml file" >&2; exit 1 ;; esac

# Unfinished TODOs: `???` is OmegaConf's "mandatory value" marker (comments are ignored)
if sed 's/#.*//' "$CONFIG" | grep -q '???'; then
    echo "submit: $CONFIG still contains ??? (unfilled TODOs):" >&2
    grep -n '???' "$CONFIG" | grep -v '^[0-9]*:\s*#' | head -10 >&2
    echo "Fill them in, then check with:" >&2
    echo "  anemoi-training config validate --config-path $(dirname "$CONFIG") --config-name $(basename "$CONFIG" .yaml)" >&2
    exit 1
fi

OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) OVERRIDES+=("diagnostics.log.mlflow.run_name=$2"); shift 2 ;;
        --name=*) OVERRIDES+=("diagnostics.log.mlflow.run_name=${1#--name=}"); shift ;;
        *) OVERRIDES+=("$1"); shift ;;
    esac
done

# Exercise label from the directory (exercises/<option>/[variants/]file.yaml), for the job name only
OPT_DIR="$(dirname "$CONFIG")"
[[ "$(basename "$OPT_DIR")" == variants ]] && OPT_DIR="$(dirname "$OPT_DIR")"
OPT="$(basename "$OPT_DIR" | cut -d_ -f1)"
JOB_NAME="h2-${OPT}-$(basename "$CONFIG" .yaml)"

LOG_DIR="$SCRATCH/aiwcas26/logs"
mkdir -p "$LOG_DIR"

JOB_ID=$(sbatch --parsable \
    --job-name="$JOB_NAME" \
    --output="$LOG_DIR/%x-%j.out" \
    --export=ALL,REPO_DIR="$REPO_DIR" \
    "$REPO_DIR/scripts/train.sbatch" "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"})
JOB_ID="${JOB_ID%%;*}"

echo "Submitted job $JOB_ID ($JOB_NAME)"
echo "  config : $CONFIG"
[[ ${#OVERRIDES[@]} -gt 0 ]] && echo "  extra  : ${OVERRIDES[*]}"
echo "  log    : $LOG_DIR/$JOB_NAME-$JOB_ID.out"
echo "Monitor  : squeue --me      |  tail -f $LOG_DIR/$JOB_NAME-$JOB_ID.out"
echo "Afterwards: python scripts/plot_loss.py --job $JOB_ID"
