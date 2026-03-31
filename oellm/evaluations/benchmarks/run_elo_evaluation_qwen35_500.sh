#!/bin/bash
#SBATCH --job-name=elo-q35-500
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/logs/elo_q35_500_%A_%a.log
#SBATCH --requeue
# no node exclusions

# Elo Rating estimation using Qwen3.5-27B judge (LMArena, with English)
# 500 battles/language variant — to test if more battles change the results.
# Judge runs as vllm serve background process; evaluated model uses VLLM in-process.
# Balanced: 500 battles/language × 12 EU languages (capped by availability for ro/el)
#
# Usage:
#   # Single model
#   sbatch oellm/evaluations/benchmarks/run_elo_evaluation_qwen35.sh eu-D1-90en
#
#   # All 12 EU models as array job
#   sbatch --array=0-11 oellm/evaluations/benchmarks/run_elo_evaluation_qwen35.sh

set -euo pipefail

# Configuration
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_BIN="$PROJECT_ROOT/.venv/bin"
SYMLINK_DIR="$PROJECT_ROOT/models/eval"

# Array job model mapping (0-11)
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
)

# Determine model from args or array index
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    MODEL_SYMLINK="${MODELS_ARRAY[$SLURM_ARRAY_TASK_ID]}"
else
    MODEL_SYMLINK="${1:?Usage: sbatch run_elo_evaluation_qwen35.sh <model-symlink>}"
fi

N_PER_LANGUAGE=500  # Balanced sampling: 500 battles per language

# Resolve model path from symlink
MODEL_PATH="$SYMLINK_DIR/$MODEL_SYMLINK"
if [ ! -e "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    ls -la "$SYMLINK_DIR/" 2>/dev/null || echo "  (directory does not exist)"
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

# EU languages (with English)
EU_LANGUAGES="en de es fr it pt pl nl cs ro el uk"

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
echo "Elo RATING ESTIMATION (Qwen3.5-27B judge)"
echo "=============================================="
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Model:            $MODEL_SYMLINK -> $(readlink -f "$MODEL_PATH")"
echo "Judge:            $JUDGE_HF_NAME (via vllm serve on port $VLLM_PORT)"
echo "Arena:            LMArena"
echo "N Per Language:   $N_PER_LANGUAGE"
echo "Languages:        $EU_LANGUAGES"
echo "Swap Mode:        $SWAP_MODE"
echo "=============================================="
echo ""

$VENV_PYTHON -c "import vllm; print(f'vllm version: {vllm.__version__}')"

# === Start vllm serve for judge ===
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

# Wait for health endpoint (up to 10 minutes)
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

# Cleanup vllm server on exit
trap "kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true" EXIT

# === Run Elo evaluation ===
cd "$OPENJURY_DIR"

$VENV_PYTHON judgearena/estimate_elo_ratings.py \
    --arena LMArena \
    --model "VLLM/$MODEL_PATH" \
    --judge "ChatOpenAI/$JUDGE_HF_NAME" \
    --swap_mode $SWAP_MODE \
    --truncate_all_input_chars $TRUNCATE_CHARS \
    --max_out_tokens_models $MAX_OUT_TOKENS \
    --max_out_tokens_judge $MAX_OUT_TOKENS_JUDGE \
    --provide_explanation \
    --engine_kwargs '{"gpu_memory_utilization": 0.40}' \
    \
    --languages $EU_LANGUAGES \
    --n_instructions_per_language $N_PER_LANGUAGE

echo ""
echo "=============================================="
echo "Elo Estimation Complete: $MODEL_SYMLINK (Qwen3.5-27B judge)"
echo "=============================================="
