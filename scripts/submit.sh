#!/bin/bash
# Submit one 1-GPU anemoi-training job for any config file (AIWCAS 2026 tutorial).
#
#   scripts/submit.sh <config.yaml> [--name RUN_NAME] [--time MINUTES] [hydra overrides...]
#
# Students: use `aiwcas train <config.yaml> ...` (bin/aiwcas), which validates the config first
# and then calls this script. Examples:
#   scripts/submit.sh exercises/A_processor_zoo/solution.yaml
#   scripts/submit.sh exercises/A_processor_zoo/mine.yaml --name A-gt-128ch model.num_channels=128
#
# What it does for you:
#   * refuses configs that still contain ??? (unfinished TODOs): validate first;
#   * resolves the config to an ABSOLUTE path (`train --config-path` breaks with relative paths);
#   * writes the Slurm log to $AIWCAS_LOGS (default $SCRATCH/aiwcas26/logs)/<job-name>-<jobid>.out
#     (#SBATCH lines cannot expand $SCRATCH, so the path is passed here);
#   * --time sets the Slurm time limit (default 10 min, at most 30 in the debug QOS);
#   * runs scripts/train.sbatch, which appends exercises/ runs to the room scoreboard when they finish.
# Environment: ANEMOI_VENV (default: the shared env), ANEMOI_ENV_MODE (auto|sqfs|venv), SCOREBOARD_DIR
# and everything `source setup.sh` exports are passed to the job.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 || "$1" == -h || "$1" == --help ]]; then
    sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
    echo "Fill them in, then check with:  aiwcas validate $CONFIG" >&2
    exit 1
fi

OVERRIDES=()
SBATCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --time) SBATCH_ARGS+=("--time=$2"); shift 2 ;;
        --time=*) SBATCH_ARGS+=("$1"); shift ;;
        --name) OVERRIDES+=("diagnostics.log.mlflow.run_name=$2"); shift 2 ;;
        --name=*) OVERRIDES+=("diagnostics.log.mlflow.run_name=${1#--name=}"); shift ;;
        *) OVERRIDES+=("$1"); shift ;;
    esac
done

# Exercise label from the directory (exercises/<option>/[variants/]file.yaml), for the job name only.
# Other configs (e.g. configs/04_tiny_train.yaml) keep their file name and skip the hour-2 scoreboard.
case "$CONFIG" in
    */exercises/*)
        OPT_DIR="$(dirname "$CONFIG")"
        [[ "$(basename "$OPT_DIR")" == variants ]] && OPT_DIR="$(dirname "$OPT_DIR")"
        OPT="$(basename "$OPT_DIR" | cut -d_ -f1)"
        JOB_NAME="h2-${OPT}-$(basename "$CONFIG" .yaml)" ;;
    *)  JOB_NAME="$(basename "$CONFIG" .yaml)"
        export SCOREBOARD="${SCOREBOARD:-0}" ;;
esac

LOG_DIR="${AIWCAS_LOGS:-$SCRATCH/aiwcas26/logs}"
mkdir -p "$LOG_DIR"

JOB_ID=$(sbatch --parsable \
    --job-name="$JOB_NAME" \
    --output="$LOG_DIR/%x-%j.out" \
    --export=ALL,REPO_DIR="$REPO_DIR" \
    ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"} \
    "$REPO_DIR/scripts/train.sbatch" "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"})
JOB_ID="${JOB_ID%%;*}"

echo "Submitted job $JOB_ID ($JOB_NAME)"
echo "  config : $CONFIG"
[[ ${#OVERRIDES[@]} -gt 0 ]] && echo "  extra  : ${OVERRIDES[*]}"
echo "  log    : $LOG_DIR/$JOB_NAME-$JOB_ID.out"
echo "Next:"
echo "  aiwcas log           follow it (about 30 s of silence for imports and setup, then a progress bar)"
echo "  aiwcas status        waiting (PENDING) or running?"
echo "  aiwcas plot          the loss curve, when it is done"
