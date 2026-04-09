#!/bin/bash
#SBATCH --job-name=wr-q35
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/logs/wr_q35_%A_%a.log
#SBATCH --requeue

# Winrate evaluation using Qwen3.5-27B judge (vllm serve)
# Compares our model vs baseline on m-arena-hard-EU and arena-hard.
#
# Array encodes: model_index * 2 + dataset_index
# dataset 0 = m-arena-hard-EU, dataset 1 = arena-hard
#
# Usage:
#   # Single model, both benchmarks
#   sbatch --array=0-1 oellm/evaluations/benchmarks/run_evaluation_qwen35.sh eu-D1-90en
#
#   # All models (23 models x 2 datasets = 46 jobs)
#   sbatch --array=0-45 oellm/evaluations/benchmarks/run_evaluation_qwen35.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_BIN="$PROJECT_ROOT/.venv/bin"
SYMLINK_DIR="$PROJECT_ROOT/models/eval"

# 23 models
MODELS_ARRAY=(
    "eu-A1-90en-step238"
    "eu-A2-80en-step228"
    "eu-A3-70en-step216"
    "eu-B1-90en"
    "eu-B2-80en"
    "eu-C0-100en"
    "eu-D1-90en"
    "eu-D2-80en"
    "eu-D3-70en"
    "eu-E1-90en"
    "eu-E2-80en"
    "eu-E3-70en"
    "eu-F1-100en"
    "eu-F2-75en"
    "eu-F3-50en"
    "eu-F4-25en"
    "eu-F5-0en"
    "eu-G1-100en"
    "eu-G2-75en"
    "eu-G3-50en"
    "eu-G4-25en"
    "eu-G5-0en"
    "instruct-v2-step3252"
)

DATASETS_ARRAY=("m-arena-hard-EU" "arena-hard")

# Determine model from args or array index
if [ -n "${1:-}" ]; then
    # Single model mode: arg is model symlink, array 0/1 picks dataset
    MODEL_SYMLINK="$1"
    DATASET_IDX="${SLURM_ARRAY_TASK_ID:-0}"
    DATASET="${DATASETS_ARRAY[$DATASET_IDX]}"
else
    # Full array mode: model_index * 2 + dataset_index
    TASK_ID="${SLURM_ARRAY_TASK_ID:?Must run as array job or pass model symlink}"
    MODEL_IDX=$((TASK_ID / 2))
    DATASET_IDX=$((TASK_ID % 2))
    MODEL_SYMLINK="${MODELS_ARRAY[$MODEL_IDX]}"
    DATASET="${DATASETS_ARRAY[$DATASET_IDX]}"
fi

# Resolve model path
MODEL_PATH="$SYMLINK_DIR/$MODEL_SYMLINK"
if [ ! -e "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Baseline
BASELINE="$PROJECT_ROOT/models/baselines/Olmo-3-7B-Instruct-SFT"
if [ ! -d "$BASELINE" ]; then
    echo "Error: Baseline not found at $BASELINE"
    exit 1
fi

# Judge model: Qwen3.5-27B served via vllm serve
JUDGE_HF_NAME="Qwen/Qwen3.5-27B"
VLLM_PORT=$((8100 + RANDOM % 900))

# Hyperparameters
MAX_OUT_TOKENS=8192
MAX_OUT_TOKENS_JUDGE=8192
TRUNCATE_CHARS=8192
SWAP_MODE="both"

# Environment
export OPENJURY_DATA="$PROJECT_ROOT/data/openjury-eval-data"
export PYTHONPATH="$OPENJURY_DIR:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

echo ""
echo "=============================================="
echo "WINRATE EVALUATION (Qwen3.5-27B judge)"
echo "=============================================="
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Model:            $MODEL_SYMLINK"
echo "Baseline:         $BASELINE"
echo "Dataset:          $DATASET"
echo "Judge:            $JUDGE_HF_NAME (port $VLLM_PORT)"
echo "=============================================="
echo ""

# Start vllm serve for judge
echo "=== Starting vllm serve for judge model ==="
$VENV_BIN/vllm serve "$JUDGE_HF_NAME" \
    --host 127.0.0.1 --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.50 \
    --enforce-eager \
    --enable-prefix-caching &
VLLM_PID=$!
echo "vllm serve PID: $VLLM_PID"

# Wait for health endpoint
echo "Waiting for vllm server to be ready..."
for i in $(seq 1 1200); do
    STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "vllm server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vllm serve process died"
        wait $VLLM_PID
        exit 1
    fi
    sleep 1
done

if [ "$STATUS" != "200" ]; then
    echo "ERROR: vllm server did not become ready within 1200s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

trap "kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true" EXIT

# Create job-specific symlinks to avoid race conditions
JOB_SUFFIX="${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
TEMP_BASELINE="$SYMLINK_DIR/baseline-wr-${JOB_SUFFIX}"
TEMP_OURS="$SYMLINK_DIR/ours-wr-${JOB_SUFFIX}"
ln -sf "$BASELINE" "$TEMP_BASELINE"
ln -sf "$(readlink -f "$MODEL_PATH")" "$TEMP_OURS"
trap "rm -f '$TEMP_BASELINE' '$TEMP_OURS'; kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true" EXIT

# Run winrate evaluation
cd "$OPENJURY_DIR"

$VENV_PYTHON judgearena/generate_and_evaluate.py \
    --dataset "$DATASET" \
    --model_A "VLLM/$TEMP_BASELINE" \
    --model_B "VLLM/$TEMP_OURS" \
    --judge_model "ChatOpenAI/$JUDGE_HF_NAME" \
    --n_instructions 8000 \
    --swap_mode $SWAP_MODE \
    --truncate_all_input_chars $TRUNCATE_CHARS \
    --max_out_tokens_models $MAX_OUT_TOKENS \
    --max_out_tokens_judge $MAX_OUT_TOKENS_JUDGE \
    --provide_explanation \
    --engine_kwargs '{"gpu_memory_utilization": 0.40}'

echo ""
echo "=============================================="
echo "Winrate Complete: $MODEL_SYMLINK / $DATASET (Qwen3.5-27B judge)"
echo "=============================================="
