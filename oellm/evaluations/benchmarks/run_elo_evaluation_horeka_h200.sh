#!/bin/bash
#SBATCH --job-name=elo-eval
#SBATCH --partition=accelerated-h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/fr_ff1042-oellm/open-instruct/oellm/evaluations/logs/elo_%A_%a.log
#SBATCH --error=/hkfs/work/workspace/scratch/fr_ff1042-oellm/open-instruct/oellm/evaluations/logs/elo_%A_%a.err
#SBATCH --account=hk-project-p0024002

# ELO Rating estimation on HoreKa H200 (141GB VRAM)
# Same as run_elo_evaluation.sh but adapted for HoreKa paths/modules.
#
# Prerequisites (transfer from kislurm):
#   1. OpenJury code:  oellm/evaluations/benchmarks/OpenJury/
#   2. Eval data:      data/openjury-eval-data/
#   3. Model checkpoint: checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-eu-A1-90en-hf/
#   4. Judge model:    models/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/
#
# Usage:
#   sbatch oellm/evaluations/benchmarks/run_elo_evaluation_horeka_h200.sh eu-A1-90en-step238 50
#   sbatch oellm/evaluations/benchmarks/run_elo_evaluation_horeka_h200.sh eu-A1-90en-step238
#   sbatch --array=0-2 oellm/evaluations/benchmarks/run_elo_evaluation_horeka_h200.sh

set -euo pipefail

# === HoreKa workspace paths ===
WS="/hkfs/work/workspace/scratch/fr_ff1042-oellm"
PROJECT_ROOT="${WS}/open-instruct"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"

# === Modules ===
module purge
module use /software/easybuild/modules/all/
module load Python/3.12.3-GCCcore-13.3.0
module load devel/cuda/12.4

source "$PROJECT_ROOT/.venv/bin/activate"

# === Model selection ===
MODELS_ARRAY=(
    "eu-A1-90en-step238"
    "eu-A2-80en-step228"
    "eu-A3-70en-step216"
)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    MODEL_SYMLINK="${MODELS_ARRAY[$SLURM_ARRAY_TASK_ID]}"
else
    MODEL_SYMLINK="${1:?Usage: sbatch run_elo_evaluation_horeka_h200.sh <model-symlink> [n_instructions]}"
fi

N_INSTRUCTIONS="${2:-}"

# Resolve model path from symlink
SYMLINK_DIR="$PROJECT_ROOT/models/eval"
MODEL_PATH="$SYMLINK_DIR/$MODEL_SYMLINK"
if [ ! -e "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Available models in $SYMLINK_DIR:"
    ls -la "$SYMLINK_DIR/" 2>/dev/null || echo "  (directory does not exist)"
    exit 1
fi

# Judge model
JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# Hyperparameters
MAX_OUT_TOKENS=32768
MAX_OUT_TOKENS_JUDGE=32768
TRUNCATE_CHARS=32768
SWAP_MODE="fixed"

# Filter to EU languages (same as m-arena-hard-EU)
EU_LANGUAGES="en de es fr it pt pl nl cs ro el uk"

# Environment
export OPENJURY_DATA="$PROJECT_ROOT/data/openjury-eval-data"
export PYTHONPATH="$OPENJURY_DIR:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

# Build n_instructions flag
N_INSTR_FLAG=""
if [ -n "$N_INSTRUCTIONS" ]; then
    N_INSTR_FLAG="--n_instructions $N_INSTRUCTIONS"
fi

mkdir -p "$PROJECT_ROOT/oellm/evaluations/logs"

echo ""
echo "=============================================="
echo "ELO RATING ESTIMATION (HoreKa H200)"
echo "=============================================="
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Model:            $MODEL_SYMLINK -> $(readlink -f "$MODEL_PATH")"
echo "Judge:            $JUDGE_MODEL"
echo "Arena:            LMArena"
echo "N Instructions:   ${N_INSTRUCTIONS:-all}"
echo "Languages:        $EU_LANGUAGES"
echo "Swap Mode:        $SWAP_MODE"
echo "=============================================="
echo ""

cd "$OPENJURY_DIR"

python openjury/estimate_elo_ratings.py \
    --arena LMArena \
    --model "VLLM/$MODEL_PATH" \
    --judge "VLLM/$JUDGE_MODEL" \
    --swap_mode $SWAP_MODE \
    --truncate_all_input_chars $TRUNCATE_CHARS \
    --max_out_tokens_models $MAX_OUT_TOKENS \
    --max_out_tokens_judge $MAX_OUT_TOKENS_JUDGE \
    --provide_explanation \
    --ignore_cache \
    --languages $EU_LANGUAGES \
    $N_INSTR_FLAG

echo ""
echo "=============================================="
echo "ELO Estimation Complete: $MODEL_SYMLINK"
echo "=============================================="
