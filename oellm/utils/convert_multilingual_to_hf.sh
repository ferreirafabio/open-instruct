#!/bin/bash
#SBATCH --job-name=convert-hf-multilingual
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --array=0-3
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/utils/logs/convert_hf_%A_%a.log

# Convert OLMo-core checkpoints to HuggingFace format for B1/B2/C0/D1 experiments.
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
EXPERIMENTS=("B1-90en" "B2-80en" "C0-100en" "D1-90en")
STEPS=(1290 1236 246 136)

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

echo "Conversion complete: $EXP"
ls -la "$OUTPUT"
