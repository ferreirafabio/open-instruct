#!/bin/bash
#SBATCH --job-name=convert-hf-multilingual
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --array=0-1
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/utils/logs/convert_hf_%A_%a.log

# Convert OLMo-core checkpoints to HuggingFace format.
# Runs on CPU partition (no GPU needed with --skip-validation).
#
# Usage: sbatch oellm/utils/convert_multilingual_to_hf.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OLMOCORE_PATH="/work/dlclarge2/ferreira-oellm/OLMo-core"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
CONVERT_SCRIPT="$OLMOCORE_PATH/src/examples/huggingface/convert_checkpoint_to_hf.py"

export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"

CKPT_BASE="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"

# Array: experiment name, final step
EXPERIMENTS=("E2-80en" "E3-70en")
STEPS=(710 740)

EXP="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"

INPUT="$CKPT_BASE/dolci-instruct-eu-${EXP}/step${STEP}"
OUTPUT="$CKPT_BASE/dolci-instruct-eu-${EXP}-hf"

echo ""
echo "=============================================="
echo "CHECKPOINT CONVERSION: $EXP"
echo "=============================================="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Step:   $STEP"
echo "=============================================="
echo ""

# Remove output if it exists as a file (from previous failed attempts)
if [ -f "$OUTPUT" ]; then
    rm "$OUTPUT"
fi

$VENV_PYTHON "$CONVERT_SCRIPT" \
    -i "$INPUT" \
    -o "$OUTPUT" \
    --tokenizer "$TOKENIZER" \
    --dtype bfloat16 \
    --skip-validation

# Fix max_position_embeddings (conversion sets it to null, vllm needs 65536)
CONFIG_FILE="$OUTPUT/config.json"
if [ -f "$CONFIG_FILE" ]; then
    $VENV_PYTHON -c "
import json
with open('$CONFIG_FILE') as f:
    c = json.load(f)
if c.get('max_position_embeddings') is None:
    c['max_position_embeddings'] = 65536
    with open('$CONFIG_FILE', 'w') as f:
        json.dump(c, f, indent=2)
    print('Fixed max_position_embeddings -> 65536')
else:
    print(f'max_position_embeddings already set: {c[\"max_position_embeddings\"]}')
"
fi

echo "Conversion complete: $EXP"
ls -la "$OUTPUT"
