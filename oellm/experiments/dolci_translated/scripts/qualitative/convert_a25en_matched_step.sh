#!/bin/bash
#SBATCH --job-name=convert-a25m
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/dolci_translated/logs/convert_a25m_%j.log

# Convert a single A-25en-matched OLMo-core checkpoint to HF format. Called
# incrementally as training emits each saved step — avoids the
# "wait for training, then batch convert" serialization.
#
# Usage:
#   STEP=500 sbatch oellm/experiments/dolci_translated/scripts/qualitative/convert_a25en_matched_step.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OLMOCORE_PATH="/work/dlclarge2/ferreira-oellm/OLMo-core"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
CONVERT_SCRIPT="$OLMOCORE_PATH/src/examples/huggingface/convert_checkpoint_to_hf.py"

export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"

TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"
STEP="${STEP:?STEP=<int> required}"

CKPT_BASE="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-translated-A-25en-matched"
FINAL_A75_HF="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-translated-A-75en-hf"
INPUT="$CKPT_BASE/step${STEP}"
OUTPUT="$CKPT_BASE-step${STEP}-hf"

if [ ! -d "$INPUT" ]; then
    echo "ERROR: step${STEP} not saved yet ($INPUT does not exist)"
    exit 2
fi
if [ -d "$OUTPUT" ] && [ -f "$OUTPUT/config.json" ]; then
    echo "Already converted: $OUTPUT"
    exit 0
fi

echo "Converting A-25en-matched step${STEP}..."
$VENV_PYTHON "$CONVERT_SCRIPT" -i "$INPUT" -o "$OUTPUT" \
    --tokenizer "$TOKENIZER" --dtype bfloat16 --skip-validation

# Fix max_position_embeddings if null
$VENV_PYTHON -c "
import json
with open('$OUTPUT/config.json') as f: c = json.load(f)
if c.get('max_position_embeddings') is None:
    c['max_position_embeddings'] = 65536
    with open('$OUTPUT/config.json', 'w') as f: json.dump(c, f, indent=2)
    print('Fixed max_position_embeddings')
"
# Copy chat template from A-75en final HF (same tokenizer/template)
if [ ! -f "$OUTPUT/chat_template.jinja" ] && [ -f "$FINAL_A75_HF/chat_template.jinja" ]; then
    cp "$FINAL_A75_HF/chat_template.jinja" "$OUTPUT/chat_template.jinja"
    echo "Copied chat_template.jinja"
fi

echo "Done: A-25en-matched step${STEP} -> $OUTPUT"
