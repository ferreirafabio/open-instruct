#!/bin/bash
#SBATCH --job-name=test-q35-27b
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/logs/test_qwen35_27b_%j.log
#SBATCH --exclude=dlc2gpu18,dlc2gpu19

# Smoke test: Qwen3.5-27B as judge via vllm serve + ChatOpenAI provider.
# No OpenJury code changes needed — ChatOpenAI already supports OpenAI-compatible APIs.
#
# Architecture:
#   1. Start `vllm serve Qwen3.5-27B` as background process (45% GPU)
#   2. Evaluated model loads via VLLM provider (in-process, uses remaining GPU)
#   3. Judge calls go to vllm server via ChatOpenAI (HTTP, no CUDA conflict)
#
# Usage:
#   sbatch oellm/evaluations/benchmarks/test_qwen35_27b.sh

set -euo pipefail

# Configuration
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_BIN="$PROJECT_ROOT/.venv/bin"
SYMLINK_DIR="$PROJECT_ROOT/models/eval"

# Test with D1 (best performing model)
MODEL_SYMLINK="eu-D1-90en"
MODEL_PATH="$SYMLINK_DIR/$MODEL_SYMLINK"

# Judge model
JUDGE_HF_NAME="Qwen/Qwen3.5-27B"
VLLM_PORT=$((8100 + RANDOM % 900))

# Hyperparameters (same as production)
MAX_OUT_TOKENS=8192
MAX_OUT_TOKENS_JUDGE=8192
TRUNCATE_CHARS=8192
SWAP_MODE="both"

# Environment
export OPENJURY_DATA="$PROJECT_ROOT/data/openjury-eval-data"
export PYTHONPATH="$OPENJURY_DIR:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

# Point ChatOpenAI to local vllm server
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
# Bypass proxy for localhost connections
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
# Force CUDA device visibility
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "=============================================="
echo "SMOKE TEST: Qwen3.5-27B Judge (server mode)"
echo "=============================================="
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Model:            $MODEL_SYMLINK -> $(readlink -f "$MODEL_PATH")"
echo "Judge:            $JUDGE_HF_NAME (via vllm serve on port $VLLM_PORT)"
echo "Judge provider:   ChatOpenAI → $OPENAI_BASE_URL"
echo "Arena:            LMArena"
echo "Test:             5 battles, German only"
echo "Swap Mode:        $SWAP_MODE"
echo "=============================================="
echo ""

# Use system CUDA toolkit for flashinfer JIT compilation (not Python-bundled one)
source /etc/cuda_env
cuda12.6
echo "CUDA toolkit: $(nvcc --version | tail -1)"

# Clear flashinfer JIT cache to force fresh compilation on GPU node
rm -rf /home/ferreira/.cache/flashinfer/
echo "Cleared flashinfer JIT cache"

$VENV_PYTHON -c "import vllm; print(f'vllm version: {vllm.__version__}')"

# === Start vllm serve for judge ===
echo "=== Starting vllm serve for judge model ==="
$VENV_BIN/vllm serve "$JUDGE_HF_NAME" \
    --host 127.0.0.1 --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.40 \
    --enforce-eager \
    --enable-prefix-caching &
VLLM_PID=$!
echo "vllm serve PID: $VLLM_PID"

# Wait for health endpoint (up to 10 minutes for first-time model download)
echo "Waiting for vllm server to be ready..."
for i in $(seq 1 600); do
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
    echo "ERROR: vllm server did not become ready within 600s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Cleanup vllm server on exit
trap "kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true" EXIT

# === Run Elo evaluation ===
cd "$OPENJURY_DIR"

# Note: judge uses ChatOpenAI/ prefix → routes to OPENAI_BASE_URL (our vllm server)
# Evaluated model gets 50% GPU (judge server already uses 45%)
$VENV_PYTHON judgearena/estimate_elo_ratings.py \
    --arena LMArena \
    --model "VLLM/$MODEL_PATH" \
    --judge "ChatOpenAI/$JUDGE_HF_NAME" \
    --swap_mode $SWAP_MODE \
    --truncate_all_input_chars $TRUNCATE_CHARS \
    --max_out_tokens_models $MAX_OUT_TOKENS \
    --max_out_tokens_judge $MAX_OUT_TOKENS_JUDGE \
    --provide_explanation \
    --engine_kwargs '{"gpu_memory_utilization": 0.45}' \
    \
    --languages de \
    --n_instructions_per_language 5

echo ""
echo "=============================================="
echo "SMOKE TEST PASSED: Qwen3.5-27B works as judge"
echo "=============================================="
