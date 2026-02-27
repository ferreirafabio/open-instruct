#!/bin/bash
#SBATCH --job-name=convert-v2-curve
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --array=0-18
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/think_v2_checkpoint_eval/logs/convert_%A_%a.log

# Convert OLMo-core checkpoints to HuggingFace format (SLURM array job)
# 19 tasks: steps 500-38000 (step42856 already converted as dolci-think-sft-v2-horeka-hf)

set -euo pipefail

STEPS=(500 1000 2000 3000 4000 5000 7000 8000 11000 13000 15000 17000 19000 21000 24000 27000 31000 34000 38000)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
CKPT_ROOT="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft"
OLMOCORE_PATH="/work/dlclarge2/ferreira-oellm/OLMo-core"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
CONVERT_SCRIPT="$OLMOCORE_PATH/src/examples/huggingface/convert_checkpoint_to_hf.py"
TOKENIZER="allenai/Olmo-3-7B-Think-SFT"

export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"

# Map step to source directory
if [ "$STEP" -ge 37000 ]; then
    INPUT="$CKPT_ROOT/dolci-think-sft-v2-horeka/step${STEP}"
else
    INPUT="$CKPT_ROOT/dolci-think-sft-v2/step${STEP}"
fi

OUTPUT="$CKPT_ROOT/dolci-think-sft-v2-hf/step${STEP}"

echo "=============================================="
echo "Converting checkpoint: step${STEP}"
echo "  Input:  $INPUT"
echo "  Output: $OUTPUT"
echo "  Task:   ${SLURM_ARRAY_TASK_ID}/${#STEPS[@]}"
echo "=============================================="

if [ ! -d "$INPUT" ]; then
    echo "Error: Input checkpoint not found: $INPUT"
    exit 1
fi

# Skip if already fully converted (expect 12 files: safetensors, config, tokenizer, etc.)
if [ -d "$OUTPUT" ] && [ "$(ls "$OUTPUT" | wc -l)" -ge 10 ] && ls "$OUTPUT"/model-*.safetensors 1>/dev/null 2>&1; then
    echo "Output already exists and looks complete ($(ls "$OUTPUT" | wc -l) files), skipping: $OUTPUT"
    exit 0
fi

# Remove output if it exists from previous failed attempt
if [ -d "$OUTPUT" ]; then
    rm -rf "$OUTPUT"
fi

mkdir -p "$(dirname "$OUTPUT")"

$VENV_PYTHON "$CONVERT_SCRIPT" \
    -i "$INPUT" \
    -o "$OUTPUT" \
    --tokenizer "$TOKENIZER" \
    --dtype bfloat16 \
    --skip-validation

echo "Conversion complete for step${STEP}!"
ls -la "$OUTPUT"
