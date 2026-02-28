#!/bin/bash
#SBATCH --job-name=convert-horeka-instruct
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/utils/logs/convert_horeka_instruct_%j.log

set -euo pipefail

VENV_PYTHON="/work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/python"
OLMOCORE_PATH="/work/dlclarge2/ferreira-oellm/OLMo-core"
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"
CONVERT_SCRIPT="${OLMOCORE_PATH}/src/examples/huggingface/convert_checkpoint_to_hf.py"
INPUT="/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka/step3252"
OUTPUT="/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka-hf"
# Must specify the instruct tokenizer so the chat_template (with function-calling tags)
# is embedded in the HF checkpoint.
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"

echo "Converting HoreKa instruct checkpoint to HF format..."
echo "Input: $INPUT"
echo "Output: $OUTPUT"

# Remove output if it exists from previous failed attempt
if [ -f "$OUTPUT" ]; then
    rm "$OUTPUT"
fi

$VENV_PYTHON "$CONVERT_SCRIPT" \
    -i "$INPUT" \
    -o "$OUTPUT" \
    --tokenizer "$TOKENIZER" \
    --dtype bfloat16 \
    --skip-validation

echo "Conversion complete!"
ls -la "$OUTPUT"
