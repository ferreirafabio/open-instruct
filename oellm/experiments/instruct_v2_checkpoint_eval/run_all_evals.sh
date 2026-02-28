#!/bin/bash
# Submit all evaluation SLURM jobs for the Instruct SFT v2 training curve
# 5 checkpoints x 3 job types (winrate-fixed, winrate-both, rubric) = 15 jobs
# Jobs are batched in groups of MAX_CONCURRENT to limit GPU usage.
#
# Usage:
#   bash oellm/experiments/instruct_v2_checkpoint_eval/run_all_evals.sh
#   MAX_CONCURRENT=2 bash oellm/experiments/instruct_v2_checkpoint_eval/run_all_evals.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_single_eval.sh"

MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

STEPS=(1000 2000 3000 3252)

# Build flat list of (step, eval_mode, swap_mode) jobs
# Rubric doesn't use swap_mode, so we pass "fixed" as placeholder (ignored)
ALL_JOBS=()
for STEP in "${STEPS[@]}"; do
    ALL_JOBS+=("${STEP}:winrate:fixed")
    ALL_JOBS+=("${STEP}:winrate:both")
    ALL_JOBS+=("${STEP}:rubric:fixed")
done

TOTAL_JOBS=${#ALL_JOBS[@]}

echo "=============================================="
echo "Submitting instruct v2 evaluation jobs"
echo "  Total jobs:      $TOTAL_JOBS"
echo "  Max concurrent:  $MAX_CONCURRENT"
echo "  Checkpoints:     ${STEPS[*]}"
echo "  Benchmarks:      alpaca-eval, arena-hard, m-arena-hard-EU"
echo "=============================================="
echo ""

ALL_JOB_IDS=()
CURRENT_BATCH_IDS=()
PREV_BATCH_IDS=()
BATCH_IDX=0

for i in $(seq 0 $((TOTAL_JOBS - 1))); do
    IFS=: read -r STEP EVAL_MODE SWAP_MODE <<< "${ALL_JOBS[$i]}"

    # Job name: i2-{w|r}{f|b}-s{step}
    local_e="${EVAL_MODE:0:1}"
    local_s="${SWAP_MODE:0:1}"
    JOB_NAME="i2-${local_e}${local_s}-s${STEP}"

    SBATCH_ARGS=(--job-name="$JOB_NAME")

    # Each batch depends on the previous batch completing
    if [ ${#PREV_BATCH_IDS[@]} -gt 0 ]; then
        DEP_STR=$(IFS=:; echo "${PREV_BATCH_IDS[*]}")
        SBATCH_ARGS+=("--dependency=afterany:${DEP_STR}")
    fi

    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "$EVAL_SCRIPT" "$STEP" "$EVAL_MODE" "$SWAP_MODE" | awk '{print $NF}')
    ALL_JOB_IDS+=("$JOB_ID")
    CURRENT_BATCH_IDS+=("$JOB_ID")
    echo "  Submitted: step${STEP} ${EVAL_MODE} ${SWAP_MODE} -> job ${JOB_ID}  (batch $BATCH_IDX)"

    # When current batch is full, move to next batch
    if [ ${#CURRENT_BATCH_IDS[@]} -ge "$MAX_CONCURRENT" ]; then
        PREV_BATCH_IDS=("${CURRENT_BATCH_IDS[@]}")
        CURRENT_BATCH_IDS=()
        BATCH_IDX=$((BATCH_IDX + 1))
    fi
done

echo ""
echo "=============================================="
echo "All ${#ALL_JOB_IDS[@]} jobs submitted in $((BATCH_IDX + 1)) batches of up to $MAX_CONCURRENT"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER | grep i2-"
echo "=============================================="
