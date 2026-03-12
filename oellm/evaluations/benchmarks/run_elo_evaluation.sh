#!/bin/bash
#SBATCH --job-name=elo-eval
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/logs/elo_%A_%a.log
#SBATCH --requeue

# ELO Rating estimation using OpenJury + LMArena battles
# Estimates Bradley-Terry ELO by judging our model against arena opponents
#
# Usage:
#   # Smoke test (50 battles, single model)
#   sbatch oellm/evaluations/benchmarks/run_elo_evaluation.sh eu-A1-90en-step238 50
#
#   # Full run (all battles, single model)
#   sbatch oellm/evaluations/benchmarks/run_elo_evaluation.sh eu-A1-90en-step238
#
#   # Array job for A1/A2/A3
#   sbatch --array=0-2 oellm/evaluations/benchmarks/run_elo_evaluation.sh
#
#   # Include baseline
#   sbatch oellm/evaluations/benchmarks/run_elo_evaluation.sh instruct-baseline

set -euo pipefail

# Configuration
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
SYMLINK_DIR="$PROJECT_ROOT/models/eval"

# Array job model mapping (used when --array is set)
MODELS_ARRAY=(
    "eu-A1-90en-step238"
    "eu-A2-80en-step228"
    "eu-A3-70en-step216"
)

# Determine model from args or array index
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    MODEL_SYMLINK="${MODELS_ARRAY[$SLURM_ARRAY_TASK_ID]}"
else
    MODEL_SYMLINK="${1:?Usage: sbatch run_elo_evaluation.sh <model-symlink> [n_instructions]}"
fi

N_INSTRUCTIONS="${2:-}"  # Empty = use all battles
N_PER_LANGUAGE="${3:-200}"  # Balanced sampling: 200 battles per language

# Resolve model path from symlink
MODEL_PATH="$SYMLINK_DIR/$MODEL_SYMLINK"
if [ ! -e "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Available models in $SYMLINK_DIR:"
    ls -la "$SYMLINK_DIR/" 2>/dev/null || echo "  (directory does not exist)"
    exit 1
fi

# Judge model (same as existing evals - Qwen3-30B, NOT Qwen3.5 which is incompatible with vllm 0.10.2)
JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# Hyperparameters (consistent with run_evaluation.sh)
MAX_OUT_TOKENS=8192
MAX_OUT_TOKENS_JUDGE=8192
TRUNCATE_CHARS=8192
SWAP_MODE="both"

# Filter to EU languages (same as m-arena-hard-EU)
EU_LANGUAGES="en de es fr it pt pl nl cs ro el uk"

# Environment
export OPENJURY_DATA="$PROJECT_ROOT/data/openjury-eval-data"
export PYTHONPATH="$OPENJURY_DIR:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

# Build n_instructions flags
N_INSTR_FLAG=""
if [ -n "$N_INSTRUCTIONS" ]; then
    N_INSTR_FLAG="--n_instructions $N_INSTRUCTIONS"
fi
N_PER_LANG_FLAG=""
if [ -n "$N_PER_LANGUAGE" ]; then
    N_PER_LANG_FLAG="--n_instructions_per_language $N_PER_LANGUAGE"
fi

echo ""
echo "=============================================="
echo "ELO RATING ESTIMATION"
echo "=============================================="
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Model:            $MODEL_SYMLINK -> $(readlink -f "$MODEL_PATH")"
echo "Judge:            $JUDGE_MODEL"
echo "Arena:            LMArena"
echo "N Instructions:   ${N_INSTRUCTIONS:-all}"
echo "N Per Language:   ${N_PER_LANGUAGE:-none}"
echo "Languages:        $EU_LANGUAGES"
echo "Swap Mode:        $SWAP_MODE"
echo "=============================================="
echo ""

cd "$OPENJURY_DIR"

$VENV_PYTHON openjury/estimate_elo_ratings.py \
    --arena LMArena \
    --model "VLLM/$MODEL_PATH" \
    --judge "VLLM/$JUDGE_MODEL" \
    --swap_mode $SWAP_MODE \
    --truncate_all_input_chars $TRUNCATE_CHARS \
    --max_out_tokens_models $MAX_OUT_TOKENS \
    --max_out_tokens_judge $MAX_OUT_TOKENS_JUDGE \
    --provide_explanation \
    \
    --languages $EU_LANGUAGES \
    $N_INSTR_FLAG \
    $N_PER_LANG_FLAG

echo ""
echo "=============================================="
echo "ELO Estimation Complete: $MODEL_SYMLINK"
echo "=============================================="
