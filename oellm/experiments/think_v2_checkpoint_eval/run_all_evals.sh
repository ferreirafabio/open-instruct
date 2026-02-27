#!/bin/bash
# Submit all evaluation SLURM jobs for the Think SFT v2 training curve
# 20 checkpoints x 3 job types (winrate-fixed, winrate-both, rubric) = 60 jobs
# step42856 already has winrate-both and rubric results, only needs winrate-fixed
# Jobs are batched in groups of MAX_CONCURRENT to limit GPU usage.
#
# Usage:
#   bash oellm/experiments/think_v2_checkpoint_eval/run_all_evals.sh
#   MAX_CONCURRENT=2 bash oellm/experiments/think_v2_checkpoint_eval/run_all_evals.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_single_eval.sh"

MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

STEPS=(500 1000 2000 3000 4000 5000 7000 8000 11000 13000 15000 17000 19000 21000 24000 27000 31000 34000 38000)

# Job types: eval_mode swap_mode
# winrate runs twice (fixed + both), rubric runs once
JOB_TYPES=("winrate:fixed" "winrate:both" "rubric:fixed")

# Build flat list of (step, eval_mode, swap_mode) jobs
ALL_JOBS=()
for STEP in "${STEPS[@]}"; do
    for JT in "${JOB_TYPES[@]}"; do
        ALL_JOBS+=("${STEP}:${JT}")
    done
done
# step42856: only winrate-fixed (winrate-both and rubric already exist)
ALL_JOBS+=("42856:winrate:fixed")

TOTAL_JOBS=${#ALL_JOBS[@]}

echo "=============================================="
echo "Submitting evaluation jobs"
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

    SBATCH_ARGS=(--job-name="v2-${EVAL_MODE:0:1}${SWAP_MODE:0:1}-s${STEP}")

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
echo "  squeue -u \$USER | grep v2-"
echo "=============================================="
