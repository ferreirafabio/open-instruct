#!/bin/bash
#SBATCH --job-name=elo-datamix9b
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=06:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/logs/elo_datamix9b_%A_%a.log
#SBATCH --requeue
#SBATCH --array=0-9

# Elo eval for datamix-9b-sft (500 battles/lang)
# NOTE: model max context = 2048, so truncate_all_input_chars and
#       max_out_tokens_models are set to 1024 to stay within limit.
#
# Array mapping:
#   0-7: per-language (en, cs, de, es, fi, fr, it, sv)
#   8:   combined all languages
#   9:   combined w/o English
#
# Usage:
#   sbatch oellm/experiments/datamix_9b_sft/scripts/run_elo_per_language.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_BIN="$PROJECT_ROOT/.venv/bin"

MODEL="datamix-9b-dt-A-75en"
LANGS=(en cs de es fi fr it sv)
N_PER_LANGUAGE=500
TASK_ID="${SLURM_ARRAY_TASK_ID}"

if [ $TASK_ID -lt 8 ]; then
    LANGUAGES="${LANGS[$TASK_ID]}"
elif [ $TASK_ID -eq 8 ]; then
    LANGUAGES="en cs de es fi fr it sv"
else
    LANGUAGES="cs de es fi fr it sv"
fi

MODEL_PATH="$PROJECT_ROOT/models/eval/$MODEL"
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

echo "Task $TASK_ID | Model: $MODEL | Languages: $LANGUAGES | N=$N_PER_LANGUAGE"

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
    --engine_kwargs '{"gpu_memory_utilization": 0.40, "max_model_len": 2048}' \
    --languages $LANGUAGES \
    --n_instructions_per_language $N_PER_LANGUAGE

echo "Done: $MODEL / $LANGUAGES (N=$N_PER_LANGUAGE)"
