#!/bin/bash
#SBATCH --job-name=elo-instruct-curve
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/logs/elo_instruct_curve_%A_%a.log
#SBATCH --requeue

# ELO over training time for Instruct-SFT checkpoints
# 20k LMArena battles, all languages, caching enabled
#
# Usage:
#   sbatch --array=0-2 oellm/evaluations/benchmarks/run_elo_instruct_curve.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
SYMLINK_DIR="$PROJECT_ROOT/models/eval"

# Instruct checkpoints (baseline + step3252 already computed)
MODELS_ARRAY=(
    "instruct-v2-step1000"
    "instruct-v2-step2000"
    "instruct-v2-step3000"
)

MODEL_SYMLINK="${MODELS_ARRAY[$SLURM_ARRAY_TASK_ID]}"
N_INSTRUCTIONS=20000

MODEL_PATH="$SYMLINK_DIR/$MODEL_SYMLINK"
if [ ! -e "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_OUT_TOKENS=8192
MAX_OUT_TOKENS_JUDGE=8192
TRUNCATE_CHARS=8192
SWAP_MODE="both"

export OPENJURY_DATA="$PROJECT_ROOT/data/openjury-eval-data"
export PYTHONPATH="$OPENJURY_DIR:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

echo ""
echo "=============================================="
echo "ELO INSTRUCT CURVE: $MODEL_SYMLINK"
echo "=============================================="
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Model:            $MODEL_SYMLINK -> $(readlink -f "$MODEL_PATH")"
echo "Judge:            $JUDGE_MODEL"
echo "N Instructions:   $N_INSTRUCTIONS"
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
    --n_instructions $N_INSTRUCTIONS

echo ""
echo "ELO Complete: $MODEL_SYMLINK"
