#!/bin/bash
#SBATCH --job-name=serve-model
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/serve_%j.log

set -euo pipefail

# Configuration
VENV_PYTHON="/work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/python"
SCRIPT_DIR="/work/dlclarge2/ferreira-oellm/open-instruct/oellm"
VLLM_PORT=8000
UI_PORT=7860

# Parse arguments
MODEL_TYPE="${1:-instruct}"  # instruct or think
SHARE="${2:-}"  # pass "share" to create public link

if [ "$MODEL_TYPE" == "instruct" ]; then
    MODEL_PATH="/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf"
    MODEL_NAME="dolci-instruct"
elif [ "$MODEL_TYPE" == "think" ]; then
    MODEL_PATH="/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf"
    MODEL_NAME="dolci-think"
else
    echo "Usage: $0 [instruct|think] [share]"
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

# Build UI arguments
UI_ARGS="--api-base http://localhost:$VLLM_PORT --model $MODEL_NAME --port $UI_PORT"
if [ "$SHARE" == "share" ]; then
    UI_ARGS="$UI_ARGS --share"
    echo ""
    echo "=============================================="
    echo "PUBLIC LINK WILL BE GENERATED BELOW"
    echo "Look for: https://xxxxx.gradio.live"
    echo "=============================================="
fi

# Start Gradio UI (foreground - keeps job alive)
echo ""
echo "Starting Chat UI..."
echo "Local URL: http://$(hostname):$UI_PORT"
echo ""

$VENV_PYTHON "$SCRIPT_DIR/chat_ui.py" $UI_ARGS

# Cleanup
kill $VLLM_PID 2>/dev/null || true

