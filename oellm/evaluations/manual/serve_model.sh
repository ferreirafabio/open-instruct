#!/bin/bash
#SBATCH --job-name=serve-model
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/logs/serve_%j.log

set -euo pipefail

# Configuration
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
SCRIPT_DIR="$PROJECT_ROOT/oellm/evaluations/manual"
VLLM_PORT=8000
UI_PORT=7860

# Login node for reverse tunnel (always accessible)
LOGIN_NODE="kis3bat1.rz.ki.privat"
TUNNEL_UI_PORT=17860    # Port on login node for UI
TUNNEL_API_PORT=18000   # Port on login node for API

# Available models
declare -A MODELS
MODELS["instruct"]="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf"
MODELS["think"]="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf"
MODELS["baseline"]="$PROJECT_ROOT/models/baselines/Olmo-3-7B-Instruct-SFT"

# Parse arguments
MODEL_TYPE="${1:-instruct}"  # instruct, think, or baseline
SHARE="${2:-}"  # pass "share" to create public link

if [[ -v MODELS[$MODEL_TYPE] ]]; then
    MODEL_PATH="${MODELS[$MODEL_TYPE]}"
    MODEL_NAME="$MODEL_TYPE"
else
    echo "Usage: $0 [instruct|think|baseline] [share]"
    echo ""
    echo "Available models:"
    echo "  instruct  - Trained OLMo-3 7B instruct model (DOLCi SFT)"
    echo "  think     - Trained OLMo-3 7B think model (DOLCi SFT)"
    echo "  baseline  - Official allenai/Olmo-3-7B-Instruct-SFT (run download_baseline.sh first)"
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    if [ "$MODEL_TYPE" == "baseline" ]; then
        echo "Run: sbatch $PROJECT_ROOT/oellm/evaluations/download_instruct_baseline.sh"
    fi
    exit 1
fi

echo "=============================================="
echo "Serving model: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Node: $(hostname)"
echo "vLLM port: $VLLM_PORT"
echo "UI port: $UI_PORT"
echo "=============================================="

# Start vLLM in background
echo "Starting vLLM server..."
$VENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port $VLLM_PORT &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM to start..."
for i in {1..120}; do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM is ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM process died!"
        exit 1
    fi
    sleep 2
done

# Check if vLLM is actually ready
if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "vLLM failed to start within timeout"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Print access instructions
NODE=$(hostname)
echo ""
echo "=============================================="
echo "SERVER IS READY!"
echo ""
echo "To access from your local machine, run:"
echo ""
echo "  ssh -J ${LOGIN_NODE} -L 7860:localhost:${UI_PORT} -L 8000:localhost:${VLLM_PORT} ${NODE}"
echo ""
echo "Then open: http://localhost:7860"
echo ""
echo "Or add to ~/.ssh/config:"
echo ""
echo "  Host ${NODE}"
echo "      ProxyJump ${LOGIN_NODE}"
echo "      LocalForward 7860 localhost:${UI_PORT}"
echo "      LocalForward 8000 localhost:${VLLM_PORT}"
echo ""
echo "=============================================="
echo ""

# Build UI arguments
UI_ARGS="--api-base http://localhost:$VLLM_PORT --model $MODEL_NAME --port $UI_PORT"

# Start Gradio UI (foreground - keeps job alive)
echo "Starting Chat UI on ${NODE}:${UI_PORT}..."
echo ""

$VENV_PYTHON "$SCRIPT_DIR/chat_ui.py" $UI_ARGS

# Cleanup
kill $VLLM_PID 2>/dev/null || true

