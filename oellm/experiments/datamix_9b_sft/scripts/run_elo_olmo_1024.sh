#!/bin/bash
#SBATCH --job-name=elo-olmo-1024
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=06:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/logs/elo_olmo_1024_%A_%a.log
#SBATCH --requeue
#SBATCH --array=0-19

# Rerun OLMo models with same 1024/1024 token limit as datamix-9b for fair comparison.
# 2 models x 10 tasks (8 per-lang + combined + combined w/o en) = 20 jobs
#
# Usage:
#   sbatch oellm/experiments/datamix_9b_sft/scripts/run_elo_olmo_1024.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_BIN="$PROJECT_ROOT/.venv/bin"

MODELS_ARRAY=(
    "dt-A-75en"
    "instruct-v2-step3252"
)
LANGS=(en cs de es fi fr it sv)
N_PER_LANGUAGE=500
N_MODELS=${#MODELS_ARRAY[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID}"
MODEL_IDX=$((TASK_ID / 10))
LANG_IDX=$((TASK_ID % 10))

MODEL_SYMLINK="${MODELS_ARRAY[$MODEL_IDX]}"

if [ $LANG_IDX -lt 8 ]; then
    LANGUAGES="${LANGS[$LANG_IDX]}"
elif [ $LANG_IDX -eq 8 ]; then
    LANGUAGES="en cs de es fi fr it sv"
else
    LANGUAGES="cs de es fi fr it sv"
fi

MODEL_PATH="$PROJECT_ROOT/models/eval/$MODEL_SYMLINK"
if [ ! -e "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

JUDGE_HF_NAME="Qwen/Qwen3.5-27B"
VLLM_PORT=$((8100 + RANDOM % 900))

export OPENJURY_DATA="$PROJECT_ROOT/data/openjury-eval-data"
export PYTHONPATH="$OPENJURY_DIR:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

echo "Task $TASK_ID | Model: $MODEL_SYMLINK | Languages: $LANGUAGES | N=$N_PER_LANGUAGE"
echo "Using 1024/1024 token limit (matching datamix-9b eval settings)"

$VENV_BIN/vllm serve "$JUDGE_HF_NAME" \
    --host 127.0.0.1 --port $VLLM_PORT \
    --tensor-parallel-size 1 --dtype bfloat16 \
    --max-model-len 32768 --gpu-memory-utilization 0.50 \
    --enforce-eager --enable-prefix-caching &
VLLM_PID=$!

for i in $(seq 1 1200); do
    STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
    [ "$STATUS" = "200" ] && break
    ! kill -0 $VLLM_PID 2>/dev/null && exit 1
    sleep 1
done
[ "$STATUS" != "200" ] && exit 1
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

cd "$OPENJURY_DIR"
$VENV_PYTHON judgearena/estimate_elo_ratings.py \
    --arena LMArena \
    --model "VLLM/$MODEL_PATH" \
    --judge "ChatOpenAI/$JUDGE_HF_NAME" \
    --swap_mode both \
    --truncate_all_input_chars 1024 \
    --max_out_tokens_models 1024 \
    --max_out_tokens_judge 8192 \
    --provide_explanation \
    --engine_kwargs '{"gpu_memory_utilization": 0.40}' \
    --ignore_cache \
    --languages $LANGUAGES \
    --n_instructions_per_language $N_PER_LANGUAGE

echo "Done: $MODEL_SYMLINK / $LANGUAGES (1024/1024 limit)"
