#!/bin/bash
#SBATCH --job-name=convert-dt-qual
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1:30:00
#SBATCH --array=0-7
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/dolci_translated/logs/convert_qual_%A_%a.log

# Convert 8 intermediate dolci_translated checkpoints (A-75en + A-25en) to HF format
# for the qualitative completions webpage. Final ckpts (3998, 8686) are already
# converted at dolci-translated-A-{75en,25en}-hf.
#
# Usage:
#   sbatch oellm/experiments/dolci_translated/scripts/qualitative/convert_intermediate_ckpts.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OLMOCORE_PATH="/work/dlclarge2/ferreira-oellm/OLMo-core"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
CONVERT_SCRIPT="$OLMOCORE_PATH/src/examples/huggingface/convert_checkpoint_to_hf.py"

export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"

TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"

# (ratio, step) pairs — final steps (3998, 8686) excluded since they already exist as -hf
SPECS=(
    "A-75en:500"
    "A-75en:1500"
    "A-75en:2500"
    "A-75en:3500"
    "A-25en:500"
    "A-25en:2500"
    "A-25en:5000"
    "A-25en:7000"
)

SPEC="${SPECS[$SLURM_ARRAY_TASK_ID]}"
RATIO="${SPEC%%:*}"
STEP="${SPEC##*:}"

CKPT_BASE="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-translated-${RATIO}"
INPUT="$CKPT_BASE/step${STEP}"
OUTPUT="$CKPT_BASE-step${STEP}-hf"
FINAL_HF="${CKPT_BASE}-hf"

if [ -d "$OUTPUT" ] && [ -f "$OUTPUT/config.json" ]; then
    echo "Already exists: $OUTPUT, skipping conversion"
else
    echo "Converting ${RATIO} step${STEP}..."
    $VENV_PYTHON "$CONVERT_SCRIPT" -i "$INPUT" -o "$OUTPUT" \
        --tokenizer "$TOKENIZER" --dtype bfloat16 --skip-validation

    # Fix max_position_embeddings (sometimes null after conversion)
    $VENV_PYTHON -c "
import json
with open('$OUTPUT/config.json') as f:
    c = json.load(f)
if c.get('max_position_embeddings') is None:
    c['max_position_embeddings'] = 65536
    with open('$OUTPUT/config.json', 'w') as f:
        json.dump(c, f, indent=2)
    print('Fixed max_position_embeddings')
"
fi

# Ensure chat_template.jinja is present (copy from final HF dir if missing)
if [ ! -f "$OUTPUT/chat_template.jinja" ] && [ -f "$FINAL_HF/chat_template.jinja" ]; then
    cp "$FINAL_HF/chat_template.jinja" "$OUTPUT/chat_template.jinja"
    echo "Copied chat_template.jinja from $FINAL_HF"
fi

echo "Done: ${RATIO} step${STEP} -> $OUTPUT"
