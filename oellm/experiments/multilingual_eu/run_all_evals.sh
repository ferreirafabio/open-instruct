#!/bin/bash
# Submit all evaluation SLURM jobs for a multilingual EU experiment
# Jobs are batched in groups of MAX_CONCURRENT to limit GPU usage.
#
# Usage:
#   bash oellm/experiments/multilingual_eu/run_all_evals.sh A1-90en
#   bash oellm/experiments/multilingual_eu/run_all_evals.sh A1-90en "50 100 150"
#   MAX_CONCURRENT=2 bash oellm/experiments/multilingual_eu/run_all_evals.sh B1-90en

set -euo pipefail

EXPERIMENT="${1:?Usage: bash run_all_evals.sh <experiment> [steps]}"
CUSTOM_STEPS="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_single_eval.sh"
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
CKPT_ROOT="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft"

MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

# Auto-detect checkpoint steps if not provided
if [ -n "$CUSTOM_STEPS" ]; then
    read -ra STEPS <<< "$CUSTOM_STEPS"
else
    CKPT_DIR="$CKPT_ROOT/dolci-instruct-eu-${EXPERIMENT}-hf"
    if [ ! -d "$CKPT_DIR" ]; then
        echo "Error: No HF checkpoint directory found: $CKPT_DIR"
        echo "Convert checkpoints first."
        exit 1
    fi
    STEPS=()
    for d in "$CKPT_DIR"/step*/; do
        [ -d "$d" ] || continue
        step=$(basename "$d" | sed 's/step//')
        STEPS+=("$step")
    done
    # Sort numerically
    IFS=$'\n' STEPS=($(sort -n <<< "${STEPS[*]}")); unset IFS
fi

# Job types: winrate + rubric
JOB_TYPES=("winrate:fixed" "rubric:fixed")

# Build flat job list
ALL_JOBS=()
for STEP in "${STEPS[@]}"; do
    for JT in "${JOB_TYPES[@]}"; do
        ALL_JOBS+=("${STEP}:${JT}")
    done
done

TOTAL_JOBS=${#ALL_JOBS[@]}

echo "=============================================="
echo "Submitting evaluation jobs for: $EXPERIMENT"
echo "  Steps:           ${STEPS[*]}"
echo "  Total jobs:      $TOTAL_JOBS"
echo "  Max concurrent:  $MAX_CONCURRENT"
echo "=============================================="
echo ""

ALL_JOB_IDS=()
CURRENT_BATCH_IDS=()
PREV_BATCH_IDS=()
BATCH_IDX=0

for i in $(seq 0 $((TOTAL_JOBS - 1))); do
    IFS=: read -r STEP EVAL_MODE SWAP_MODE <<< "${ALL_JOBS[$i]}"

    SBATCH_ARGS=(--job-name="eu-${EVAL_MODE:0:1}${SWAP_MODE:0:1}-${EXPERIMENT}-s${STEP}")

    if [ ${#PREV_BATCH_IDS[@]} -gt 0 ]; then
        DEP_STR=$(IFS=:; echo "${PREV_BATCH_IDS[*]}")
        SBATCH_ARGS+=("--dependency=afterany:${DEP_STR}")
    fi

    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "$EVAL_SCRIPT" "$EXPERIMENT" "$STEP" "$EVAL_MODE" "$SWAP_MODE" | awk '{print $NF}')
    ALL_JOB_IDS+=("$JOB_ID")
    CURRENT_BATCH_IDS+=("$JOB_ID")
    echo "  Submitted: step${STEP} ${EVAL_MODE} ${SWAP_MODE} -> job ${JOB_ID}  (batch $BATCH_IDX)"

    if [ ${#CURRENT_BATCH_IDS[@]} -ge "$MAX_CONCURRENT" ]; then
        PREV_BATCH_IDS=("${CURRENT_BATCH_IDS[@]}")
        CURRENT_BATCH_IDS=()
        BATCH_IDX=$((BATCH_IDX + 1))
    fi
done

echo ""
echo "=============================================="
echo "All ${#ALL_JOB_IDS[@]} jobs submitted in $((BATCH_IDX + 1)) batches"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER | grep eu-"
echo "=============================================="
