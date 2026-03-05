#!/bin/bash
#SBATCH --job-name=v2-curve-eval
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/think_v2_checkpoint_eval/logs/eval_%j.log

# Evaluate a single checkpoint from the Think SFT v2 training curve
# Runs alpaca-eval then arena-hard sequentially in one job
#
# Features:
#   - Completion detection: skips benchmarks that already have results
#   - Selective caching: deletes checkpoint cache (always fresh), preserves baseline cache
#   - SLURM requeue: preempted jobs are automatically resubmitted
#
# Usage:
#   sbatch run_single_eval.sh <step> <eval_mode> <swap_mode>
#   sbatch run_single_eval.sh 5000 winrate fixed
#   sbatch run_single_eval.sh 5000 winrate both
#   sbatch run_single_eval.sh 5000 rubric rubric

set -euo pipefail

STEP="${1:?Usage: sbatch run_single_eval.sh <step> <eval_mode> <swap_mode>}"
EVAL_MODE="${2:?Usage: sbatch run_single_eval.sh <step> <eval_mode> <swap_mode>}"
SWAP_MODE="${3:?Usage: sbatch run_single_eval.sh <step> <eval_mode> <swap_mode>}"

# Configuration
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
CKPT_ROOT="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

BASELINE="$PROJECT_ROOT/models/baselines/Olmo-3-7B-Think-SFT"

# step42856 uses a different HF checkpoint directory
if [ "$STEP" -eq 42856 ]; then
    TRAINED="$CKPT_ROOT/dolci-think-sft-v2-horeka-hf"
else
    TRAINED="$CKPT_ROOT/dolci-think-sft-v2-hf/step${STEP}"
fi

EXPERIMENT_NAME="think-v2-curve-${EVAL_MODE}-${SWAP_MODE}-step${STEP}"

# Generation hyperparameters (match run_evaluation.sh defaults)
MAX_OUT_TOKENS=32768
MAX_OUT_TOKENS_JUDGE=32768
TRUNCATE_CHARS=32768
N_INSTRUCTIONS=50000
JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

BASELINE_NAME="$(basename "$BASELINE")"

# Reuse existing result directory if found, otherwise create a new timestamped one.
# This ensures requeued jobs write into the same directory as the original attempt.
EXISTING_DIR=$(ls -d "$OPENJURY_DIR/results/${EXPERIMENT_NAME}-${BASELINE_NAME}-"*/ 2>/dev/null | head -1 || true)
if [ -n "$EXISTING_DIR" ]; then
    RESULTS_ROOT="${EXISTING_DIR%/}"
    echo "Reusing existing results directory: $RESULTS_ROOT"
else
    RESULTS_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RESULTS_ROOT="$OPENJURY_DIR/results/${EXPERIMENT_NAME}-${BASELINE_NAME}-${RESULTS_TIMESTAMP}"
fi

# Environment
export OPENJURY_DATA="$PROJECT_ROOT/data/openjury-eval-data"
export PYTHONPATH="$OPENJURY_DIR:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

# Verify models exist
if [ ! -d "$BASELINE" ]; then
    echo "Error: Baseline model not found: $BASELINE"
    exit 1
fi
if [ ! -d "$TRAINED" ]; then
    echo "Error: Trained model not found: $TRAINED"
    exit 1
fi

# Symlinks for short model names (avoids path-too-long errors in OpenJury)
SYMLINK_DIR="$PROJECT_ROOT/models/eval"
mkdir -p "$SYMLINK_DIR"

TEMP_BASELINE="$SYMLINK_DIR/think-baseline"
SYMLINK_NAME="think-v2-step${STEP}"
TEMP_OURS="$SYMLINK_DIR/$SYMLINK_NAME"
rm -f "$TEMP_BASELINE" "$TEMP_OURS"
ln -sf "$BASELINE" "$TEMP_BASELINE"
ln -sf "$TRAINED" "$TEMP_OURS"

# --- Completion detection ---
# Check if results already exist for a specific (experiment, dataset)
check_completed() {
    local dataset=$1
    for d in "$OPENJURY_DIR/results/${EXPERIMENT_NAME}-"*/; do
        [ -d "$d" ] || continue
        if ls "$d"/${dataset}-*/results-*.json 1>/dev/null 2>&1; then
            echo "SKIP: $dataset already has results in $d"
            return 0
        fi
    done
    return 1
}

# --- Cache deletion ---
# Delete both baseline and checkpoint cache files to force fresh generation.
#
# OpenJury cache_name = f"{dataset}_{model}_{n_instructions}"
# model paths are passed as "VLLM/$path", so the slashes create subdirectories:
#   cache/<dataset>_VLLM/<abs_path_to_symlink>_<n_instructions>.csv.zip
delete_model_caches() {
    local dataset=$1
    local cache_dir="$OPENJURY_DATA/cache"
    for model_path in "$TEMP_BASELINE" "$TEMP_OURS"; do
        local cache_file="$cache_dir/${dataset}_VLLM${model_path}_${N_INSTRUCTIONS}.csv.zip"
        if [ -f "$cache_file" ]; then
            echo "Deleting cache: $cache_file"
            rm -f "$cache_file"
        fi
    done
}

run_eval() {
    local dataset=$1
    echo ""
    echo "--- Evaluating: $dataset ($EVAL_MODE, swap=$SWAP_MODE) ---"
    echo ""

    cd "$OPENJURY_DIR"

    $VENV_PYTHON openjury/generate_and_evaluate.py \
        --dataset "$dataset" \
        --model_A "VLLM/$TEMP_BASELINE" \
        --model_B "VLLM/$TEMP_OURS" \
        --judge_model "VLLM/$JUDGE_MODEL" \
        --n_instructions "$N_INSTRUCTIONS" \
        --result_folder "$RESULTS_ROOT" \
        --max_out_tokens_models $MAX_OUT_TOKENS \
        --max_out_tokens_judge $MAX_OUT_TOKENS_JUDGE \
        --truncate_all_input_chars $TRUNCATE_CHARS \
        --swap_mode $SWAP_MODE \
        --eval_mode $EVAL_MODE \
        --provide_explanation \
        --ignore_cache
}

echo ""
echo "=============================================="
echo "THINK V2 CHECKPOINT EVALUATION"
echo "=============================================="
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Step:           $STEP"
echo "Eval Mode:      $EVAL_MODE"
echo "Swap Mode:      $SWAP_MODE"
echo "Experiment:     $EXPERIMENT_NAME"
echo "Baseline:       ${BASELINE#$PROJECT_ROOT/}"
echo "Trained:        ${TRAINED#$PROJECT_ROOT/}"
echo "Judge:          $JUDGE_MODEL"
echo "Results:        $RESULTS_ROOT"
echo "=============================================="
echo ""

# Determine which benchmarks still need to run
DATASETS_TO_RUN=()
for dataset in alpaca-eval arena-hard; do
    if ! check_completed "$dataset"; then
        DATASETS_TO_RUN+=("$dataset")
    fi
done

if [ ${#DATASETS_TO_RUN[@]} -eq 0 ]; then
    echo "All benchmarks already completed for $EXPERIMENT_NAME, skipping."
    exit 0
fi

echo "Benchmarks to run: ${DATASETS_TO_RUN[*]}"

# Delete all model caches for datasets we're about to run (force fresh generation)
for dataset in "${DATASETS_TO_RUN[@]}"; do
    delete_model_caches "$dataset"
done

# Run remaining benchmarks
for dataset in "${DATASETS_TO_RUN[@]}"; do
    run_eval "$dataset"
done

echo ""
echo "=============================================="
echo "Evaluation complete for step${STEP} (${EVAL_MODE}, swap=${SWAP_MODE})"
echo "Results: $RESULTS_ROOT"
echo "=============================================="
