#!/bin/bash
#SBATCH --job-name=llm-judge-eval
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/logs/eval_%j.log

# LLM-Judge evaluation using OpenJury
# Compares trained model against baseline using automated judges
#
# Usage:
#   sbatch oellm/evaluations/benchmarks/run_evaluation.sh instruct
#   sbatch oellm/evaluations/benchmarks/run_evaluation.sh think
#   sbatch oellm/evaluations/benchmarks/run_evaluation.sh instruct all
#   sbatch oellm/evaluations/benchmarks/run_evaluation.sh think arena-hard
#   sbatch oellm/evaluations/benchmarks/run_evaluation.sh think all 100
#   sbatch oellm/evaluations/benchmarks/run_evaluation.sh instruct all 50000 my-experiment
#   sbatch oellm/evaluations/benchmarks/run_evaluation.sh think all 8000 my-exp /path/to/trained/model
#   EVAL_MODE=rubric sbatch oellm/evaluations/benchmarks/run_evaluation.sh instruct alpaca-eval

set -euo pipefail

# Configuration
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Arguments
MODEL_TYPE="${1:-instruct}"       # instruct or think
DATASET="${2:-alpaca-eval}"       # alpaca-eval, arena-hard, m-arena-hard-EU, or all
N_INSTRUCTIONS="${3:-50000}"      # High number = use all available instructions
EXPERIMENT_NAME="${4:-quick-fix-65k-config}"
TRAINED_OVERRIDE="${5:-}"         # Optional: path to trained model (overrides default)

# Generation hyperparameters
MAX_OUT_TOKENS="${MAX_OUT_TOKENS:-8192}"
MAX_OUT_TOKENS_JUDGE="${MAX_OUT_TOKENS_JUDGE:-8192}"
TRUNCATE_CHARS="${TRUNCATE_CHARS:-8192}"
SWAP_MODE="${SWAP_MODE:-both}"  # "fixed" or "both" (both corrects position bias, 2x compute)
EVAL_MODE="${EVAL_MODE:-winrate}"  # "winrate" or "rubric" (rubric: independent 1-7 Likert on 4 criteria)
PROVIDE_EXPLANATION="${PROVIDE_EXPLANATION:-true}"  # Judge explains reasoning before verdict
#EXPERIMENT_NAME="${4:-baseline-repro}"  # Old experiment name

# Set models based on type
if [ "$MODEL_TYPE" == "instruct" ]; then
    BASELINE="$PROJECT_ROOT/models/baselines/Olmo-3-7B-Instruct-SFT"
    TRAINED="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka-hf"
elif [ "$MODEL_TYPE" == "think" ]; then
    BASELINE="$PROJECT_ROOT/models/baselines/Olmo-3-7B-Think-SFT"
    TRAINED="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-horeka-hf"
    #TRAINED="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf-65k-config-fix"  # Previous (65k config fix)
else
    echo "Error: MODEL_TYPE must be 'instruct' or 'think', got: $MODEL_TYPE"
    echo "Usage: sbatch run_evaluation.sh [instruct|think] [dataset] [n_instructions] [experiment_name] [trained_model_path]"
    exit 1
fi

# Override trained model if 5th argument provided
if [ -n "$TRAINED_OVERRIDE" ]; then
    TRAINED="$TRAINED_OVERRIDE"
fi

# Timestamped, informative results root to keep runs organized
RESULTS_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASELINE_NAME="$(basename "$BASELINE")"
RESULTS_ROOT="$OPENJURY_DIR/results/${EXPERIMENT_NAME}-${BASELINE_NAME}-${RESULTS_TIMESTAMP}"

# Verify models exist
if [ ! -d "$BASELINE" ]; then
    echo "Error: Baseline model not found: $BASELINE"
    echo "Download it first: sbatch oellm/evaluations/benchmarks/download_${MODEL_TYPE}_baseline.sh"
    exit 1
fi
if [ ! -d "$TRAINED" ]; then
    echo "Error: Trained model not found: $TRAINED"
    exit 1
fi

# Judge model - Qwen3-30B is recommended for quality
JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# Note: Qwen3.5-35B-A3B (qwen3_5_moe) is NOT supported by vllm 0.10.2

# Environment
export OPENJURY_DATA="$PROJECT_ROOT/data/openjury-eval-data"
export PYTHONPATH="$OPENJURY_DIR:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

# Debug file (debug_examples.txt) is always created in the results directory by OpenJury

# Always ignore cache to ensure fresh generation and avoid stale/corrupted cached outputs
IGNORE_CACHE="1"

# Print configuration summary at start
echo ""
echo "=============================================="
echo "EVALUATION CONFIGURATION"
echo "=============================================="
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Model Type:       $MODEL_TYPE"
echo "Dataset:          $DATASET"
echo "N Instructions:   $N_INSTRUCTIONS"
echo "Experiment Name:  $EXPERIMENT_NAME"
echo ""
echo "Models:"
echo "  Baseline:       ${BASELINE#$PROJECT_ROOT/}"
echo "  Trained:        ${TRAINED#$PROJECT_ROOT/}"
echo "  Judge:          $JUDGE_MODEL"
echo ""
echo "Hyperparameters:"
echo "  IGNORE_CACHE:       $IGNORE_CACHE"
echo "  eval_mode:          $EVAL_MODE"
echo "  max_out_tokens:     $MAX_OUT_TOKENS"
echo "  max_out_tokens_judge: $MAX_OUT_TOKENS_JUDGE"
echo "  truncate_chars:     $TRUNCATE_CHARS"
echo "  swap_mode:          $SWAP_MODE"
echo "  provide_explanation: $PROVIDE_EXPLANATION"
echo ""
echo "Output:"
echo "  Results Dir:    $RESULTS_ROOT"
echo "  Debug File:     $RESULTS_ROOT/debug_examples.txt"
echo "=============================================="
echo ""

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

    # Build provide_explanation flag
    EXPLAIN_FLAG=""
    if [ "$PROVIDE_EXPLANATION" == "true" ]; then
        EXPLAIN_FLAG="--provide_explanation"
    fi

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
        $EXPLAIN_FLAG \
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
    echo "Eval Mode: $EVAL_MODE"
    echo "Results Dir: $RESULTS_ROOT"
    echo "Date: $(date)"
    echo ""
    echo "--- Hyperparameters ---"
    echo "eval_mode: $EVAL_MODE"
    echo "max_out_tokens_models: $MAX_OUT_TOKENS"
    echo "max_out_tokens_judge: $MAX_OUT_TOKENS_JUDGE"
    echo "truncate_all_input_chars: $TRUNCATE_CHARS"
    echo "swap_mode: $SWAP_MODE"
    echo "provide_explanation: $PROVIDE_EXPLANATION"
    echo "n_instructions: $N_INSTRUCTIONS"
    echo ""
    echo "--- Models ---"
    echo "experiment_name: $EXPERIMENT_NAME"
    echo "baseline: $BASELINE"
    echo "trained: $TRAINED"
    echo "judge: $JUDGE_MODEL"
    echo "tokenizer: (from model, via vllm.LLM.chat())"
    echo ""

    if [ "$EVAL_MODE" == "rubric" ]; then
        echo "=============================================="
        echo "RUBRIC RESULTS"
        echo "=============================================="
        for results_json in "$RESULTS_ROOT"/*/results-*.json; do
            [ -f "$results_json" ] || continue
            echo ""
            $VENV_PYTHON -c "
import json, sys
with open('$results_json') as f:
    r = json.load(f)
print(f\"Dataset: {r['dataset']}\")
print(f\"Judge:   {r['judge_model']}\")
print(f\"Model A: {r['model_A']}\")
print(f\"Model B: {r['model_B']}\")
print()
print(f\"  {'Criterion':<25s} {'Model A':>10s} {'Model B':>10s}\")
print('  ' + '-' * 47)
for c in r['criteria']:
    label = c.replace('_', ' ').title()
    sa = r['model_A_scores'].get(f'{c}_score', 0)
    sb = r['model_B_scores'].get(f'{c}_score', 0)
    print(f'  {label:<25s} {sa:>10.2f} {sb:>10.2f}')
print('  ' + '-' * 47)
ca = r['model_A_scores'].get('composite_score', 0)
cb = r['model_B_scores'].get('composite_score', 0)
print(f\"  {'Composite (0-1)':<25s} {ca:>10.3f} {cb:>10.3f}\")
fa = r.get('model_A_parse_failures', 0)
fb = r.get('model_B_parse_failures', 0)
print(f\"  Evaluations: {r['num_instructions']} | Parse failures: A={fa}, B={fb}\")
print()
"
        done
        echo ""
    else
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
    fi

} > "$SUMMARY_FILE" 2>&1

echo ""
echo "Output files:"
echo "  Summary:    $SUMMARY_FILE"
echo "  Debug log:  $RESULTS_ROOT/debug_examples.txt"
echo "  Results:    $RESULTS_ROOT"
echo ""
echo "Quick view:"
echo "  cat $SUMMARY_FILE"

