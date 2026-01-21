#!/bin/bash
#SBATCH --job-name=convert-hf
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/utils/logs/convert_hf_%j.log

set -euo pipefail

VENV_PYTHON="/work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/python"
CONVERT_SCRIPT="/work/dlclarge2/ferreira-oellm/OLMo-core/src/examples/huggingface/convert_checkpoint_to_hf.py"

MODEL_TYPE="${1:-instruct}"  # instruct or think

if [ "$MODEL_TYPE" == "instruct" ]; then
    INPUT="/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft/step3394"
    OUTPUT="/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf"
elif [ "$MODEL_TYPE" == "think" ]; then
    INPUT="/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft/step43376"
    OUTPUT="/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf"
else
    echo "Unknown model type: $MODEL_TYPE"
    exit 1
fi

echo "Converting $MODEL_TYPE model..."
echo "Input: $INPUT"
echo "Output: $OUTPUT"

# Remove output if it exists as a file (from previous failed attempts)
if [ -f "$OUTPUT" ]; then
    rm "$OUTPUT"
fi

$VENV_PYTHON "$CONVERT_SCRIPT" \
    -i "$INPUT" \
    -o "$OUTPUT" \
    --dtype bfloat16 \
    --skip-validation

echo "Conversion complete!"
ls -la "$OUTPUT"

