#!/bin/bash
#SBATCH --job-name=llm-judge-eval
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/logs/eval_%j.log

# LLM-Judge evaluation using OpenJury
# Compares trained model against baseline using automated judges
#
# Usage:
#   sbatch run_evaluation.sh instruct              # Run instruct model on alpaca-eval
#   sbatch run_evaluation.sh think                 # Run think model on alpaca-eval
#   sbatch run_evaluation.sh instruct all          # Run instruct on all 3 datasets
#   sbatch run_evaluation.sh think arena-hard      # Run think on specific dataset
#   sbatch run_evaluation.sh think all 100         # Limit to 100 instructions (for testing)

set -euo pipefail

# Configuration
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Arguments
MODEL_TYPE="${1:-instruct}"   # instruct or think
DATASET="${2:-alpaca-eval}"   # alpaca-eval, arena-hard, m-arena-hard-EU, or all
N_INSTRUCTIONS="${3:-50000}"  # High number = use all available instructions

# Set models based on type
if [ "$MODEL_TYPE" == "instruct" ]; then
    BASELINE="$PROJECT_ROOT/models/baselines/Olmo-3-7B-Instruct-SFT"
    TRAINED="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf"
elif [ "$MODEL_TYPE" == "think" ]; then
    # Default to stronger 32B think baseline; keep old 7B baseline noted for reference
    BASELINE="$PROJECT_ROOT/models/baselines/Olmo-3-7B-Think-SFT"
    #BASELINE="$PROJECT_ROOT/models/baselines/Olmo-3-32B-Think-SFT"
    TRAINED="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf"
else
    echo "Error: MODEL_TYPE must be 'instruct' or 'think', got: $MODEL_TYPE"
    echo "Usage: sbatch run_evaluation.sh [instruct|think] [dataset] [n_instructions]"
    exit 1
fi

# Timestamped, informative results root to keep runs organized
RESULTS_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASELINE_NAME="$(basename "$BASELINE")"
RESULTS_ROOT="$OPENJURY_DIR/results/${MODEL_TYPE}-${BASELINE_NAME}-${RESULTS_TIMESTAMP}"

# Verify models exist
if [ ! -d "$BASELINE" ]; then
    echo "Error: Baseline model not found: $BASELINE"
    echo "Download it first: sbatch oellm/evaluations/download_${MODEL_TYPE}_baseline.sh"
    exit 1
fi
if [ ! -d "$TRAINED" ]; then
    echo "Error: Trained model not found: $TRAINED"
    exit 1
fi

# Judge model - Qwen3-30B is recommended for quality
JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# Environment
export OPENJURY_EVAL_DATA="$PROJECT_ROOT/data/openjury-eval-data"
export PYTHONPATH="$OPENJURY_DIR:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

# Debug file is always created in the results directory
# Clear any old debug file at job start so all datasets accumulate fresh
DEBUG_FILE="$RESULTS_ROOT/debug_${SLURM_JOB_ID:-local}.txt"
rm -f "$DEBUG_FILE" 2>/dev/null || true
export DEBUG_EVAL_FILE="$DEBUG_FILE"

# Ignore cache - set IGNORE_CACHE=1 to force fresh generation (avoids stale cached outputs)
# Usage: IGNORE_CACHE=1 sbatch run_evaluation.sh ...
IGNORE_CACHE="${IGNORE_CACHE:-0}"

# Function to run single evaluation
run_eval() {
    local dataset=$1
    echo ""
    echo "=============================================="
    echo "LLM-Judge Evaluation: $MODEL_TYPE / $dataset"
    echo "=============================================="
    echo "Models:"
    echo "  Baseline: ${BASELINE#$PROJECT_ROOT/}"
    echo "  Ours:     ${TRAINED#$PROJECT_ROOT/}"
    echo "Judge: $JUDGE_MODEL"
    echo "Instructions: $N_INSTRUCTIONS"
    echo "=============================================="
    echo ""

    cd "$OPENJURY_DIR"
    
    # Use short, type-prefixed names for result filenames to:
    # 1. Avoid path-too-long errors
    # 2. Keep instruct and think results separate
    SYMLINK_DIR="$PROJECT_ROOT/models/eval"
    mkdir -p "$SYMLINK_DIR"
    
    TEMP_BASELINE="$SYMLINK_DIR/${MODEL_TYPE}-baseline"
    TEMP_OURS="$SYMLINK_DIR/${MODEL_TYPE}-ours"
    rm -f "$TEMP_BASELINE" "$TEMP_OURS"
    ln -sf "$BASELINE" "$TEMP_BASELINE"
    ln -sf "$TRAINED" "$TEMP_OURS"
    
    # Build optional flags
    EXTRA_FLAGS=""
    if [ "$IGNORE_CACHE" == "1" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --ignore_cache"
        echo "Note: --ignore_cache enabled, forcing fresh generation"
    fi

    $VENV_PYTHON openjury/generate_and_evaluate.py \
        --dataset "$dataset" \
        --model_A "VLLM/$TEMP_BASELINE" \
        --model_B "VLLM/$TEMP_OURS" \
        --judge_model "VLLM/$JUDGE_MODEL" \
        --n_instructions "$N_INSTRUCTIONS" \
        --result_folder "$RESULTS_ROOT" \
        $EXTRA_FLAGS
}

# Run evaluation(s)
if [ "$DATASET" == "all" ]; then
    echo "Running all 3 datasets..."
    run_eval "alpaca-eval"
    run_eval "arena-hard"
    run_eval "m-arena-hard-EU"
else
    run_eval "$DATASET"
fi

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "=============================================="

# Generate summary file in the results directory
SUMMARY_FILE="$RESULTS_ROOT/summary.txt"
echo "Generating summary file: $SUMMARY_FILE"

{
    echo "=============================================="
    echo "EVALUATION SUMMARY"
    echo "=============================================="
    echo "Job ID: ${SLURM_JOB_ID:-local}"
    echo "Model Type: $MODEL_TYPE"
    echo "Results Dir: $RESULTS_ROOT"
    echo "Date: $(date)"
    echo ""
    
    echo "=============================================="
    echo "PREFERENCE SUMMARY (summarize_preferences.py)"
    echo "=============================================="
    $VENV_PYTHON "$PROJECT_ROOT/oellm/evaluations/benchmarks/summarize_preferences.py" --results-dir "$RESULTS_ROOT"
    echo ""
    
    echo "=============================================="
    echo "WINRATE TABLE (analyse_results.py)"
    echo "=============================================="
    $VENV_PYTHON "$PROJECT_ROOT/oellm/evaluations/benchmarks/analyse_results.py" --results-dir "$RESULTS_ROOT"
    echo ""
    
    echo "=============================================="
    echo "ORIGINAL FORMAT (analyse_results_original.py)"
    echo "=============================================="
    $VENV_PYTHON "$PROJECT_ROOT/oellm/evaluations/benchmarks/analyse_results_original.py" --results-dir "$RESULTS_ROOT" 2>/dev/null || echo "(skipped - may not support --results-dir)"
    echo ""
    
} > "$SUMMARY_FILE" 2>&1

echo ""
echo "Output files:"
echo "  Summary:    $SUMMARY_FILE"
echo "  Debug log:  $DEBUG_FILE"
echo "  Results:    $RESULTS_ROOT"
echo ""
echo "Quick view:"
echo "  cat $SUMMARY_FILE"

