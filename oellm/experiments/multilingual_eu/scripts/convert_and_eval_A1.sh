#!/bin/bash
#SBATCH --job-name=convert-eval-A1
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/multilingual_eu/logs/convert_eval_A1_%j.log

# Convert A1-90en final checkpoint to HF format, then evaluate against baseline.

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OLMOCORE_PATH="/work/dlclarge2/ferreira-oellm/OLMo-core"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
CONVERT_SCRIPT="${OLMOCORE_PATH}/src/examples/huggingface/convert_checkpoint_to_hf.py"

export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

EXPERIMENT="A1-90en"
CKPT_ROOT="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"

# ---- Step 1: Convert to HF ----
echo "=============================================="
echo "STEP 1: Convert OLMo-core -> HF"
echo "=============================================="

for STEP_DIR in "$CKPT_ROOT/dolci-instruct-eu-${EXPERIMENT}"/step*; do
    STEP=$(basename "$STEP_DIR" | sed 's/step//')
    OUTPUT="$CKPT_ROOT/dolci-instruct-eu-${EXPERIMENT}-hf/step${STEP}"

    if [ -d "$OUTPUT" ] && [ "$(ls -A "$OUTPUT" 2>/dev/null)" ]; then
        echo "[step${STEP}] Already converted, skipping."
        continue
    fi

    echo ""
    echo "Converting step${STEP}..."
    echo "  Input:  $STEP_DIR"
    echo "  Output: $OUTPUT"

    $VENV_PYTHON "$CONVERT_SCRIPT" \
        -i "$STEP_DIR" \
        -o "$OUTPUT" \
        --tokenizer "$TOKENIZER" \
        --dtype bfloat16 \
        --skip-validation

    echo "[step${STEP}] Done!"
done

echo ""
echo "All conversions complete!"
ls -d "$CKPT_ROOT/dolci-instruct-eu-${EXPERIMENT}-hf"/step*

# ---- Step 2: Evaluate final checkpoint ----
echo ""
echo "=============================================="
echo "STEP 2: Evaluate (m-arena-hard-EU + arena-hard)"
echo "=============================================="

# Find the final step
FINAL_STEP=$(ls -d "$CKPT_ROOT/dolci-instruct-eu-${EXPERIMENT}-hf"/step* | sort -t'p' -k2 -n | tail -1 | xargs basename | sed 's/step//')
echo "Final step: $FINAL_STEP"

# Run evaluation
cd "$PROJECT_ROOT"
bash oellm/experiments/multilingual_eu/run_single_eval.sh "$EXPERIMENT" "$FINAL_STEP" winrate fixed

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "=============================================="
