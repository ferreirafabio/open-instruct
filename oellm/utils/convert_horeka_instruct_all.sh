#!/bin/bash
#SBATCH --job-name=convert-instruct-all
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/utils/logs/convert_instruct_all_%j.log

# Convert all instruct OLMo-core checkpoints to HF format
# Output structure matches openeurollm/OLMo-3-7B-Think-SFT: step{N}/ subdirectories

set -euo pipefail

VENV_PYTHON="/work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/python"
OLMOCORE_PATH="/work/dlclarge2/ferreira-oellm/OLMo-core"
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"
export HF_HOME="/work/dlclarge2/ferreira-oellm/open-instruct/models/huggingface"
CONVERT_SCRIPT="${OLMOCORE_PATH}/src/examples/huggingface/convert_checkpoint_to_hf.py"

CKPT_DIR="/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka"
HF_DIR="/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka-hf-all"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"

STEPS=(1000 2000 3000 3150 3252)

mkdir -p "$HF_DIR"

for step in "${STEPS[@]}"; do
    INPUT="$CKPT_DIR/step${step}"
    OUTPUT="$HF_DIR/step${step}"

    if [ -d "$OUTPUT" ] && [ "$(ls -A "$OUTPUT" 2>/dev/null)" ]; then
        echo "[step${step}] Already converted, skipping."
        continue
    fi

    echo ""
    echo "=============================================="
    echo "Converting step${step}..."
    echo "=============================================="
    echo "Input:  $INPUT"
    echo "Output: $OUTPUT"

    rm -rf "$OUTPUT"

    $VENV_PYTHON "$CONVERT_SCRIPT" \
        -i "$INPUT" \
        -o "$OUTPUT" \
        --tokenizer "$TOKENIZER" \
        --dtype bfloat16 \
        --skip-validation

    echo "[step${step}] Done!"
    ls "$OUTPUT"
    echo ""
done

echo ""
echo "All conversions complete!"
echo "Output: $HF_DIR"
du -sh "$HF_DIR"/step*
