#!/bin/bash
# Upload all 20 HF checkpoints to openeurollm HuggingFace org
# Run from login node or as SLURM job (needs network access)
#
# Usage:
#   bash oellm/experiments/think_v2_checkpoint_eval/upload_to_hf.sh
#   bash oellm/experiments/think_v2_checkpoint_eval/upload_to_hf.sh --dry-run

set -euo pipefail

DRY_RUN=false
if [ "${1:-}" == "--dry-run" ]; then
    DRY_RUN=true
    echo "DRY RUN - will not upload anything"
fi

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
CKPT_ROOT="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft"
HF_ORG="openeurollm"

export HF_HOME="$PROJECT_ROOT/models/huggingface"

STEPS=(500 1000 2000 3000 4000 5000 7000 8000 11000 13000 15000 17000 19000 21000 24000 27000 31000 34000 38000 42856)

echo "=============================================="
echo "Uploading ${#STEPS[@]} checkpoints to ${HF_ORG}"
echo "=============================================="

for STEP in "${STEPS[@]}"; do
    # step42856 lives in a different directory
    if [ "$STEP" -eq 42856 ]; then
        LOCAL_PATH="$CKPT_ROOT/dolci-think-sft-v2-horeka-hf"
    else
        LOCAL_PATH="$CKPT_ROOT/dolci-think-sft-v2-hf/step${STEP}"
    fi

    REPO_ID="${HF_ORG}/OLMo-3-7B-Think-SFT-v2"
    PATH_IN_REPO="step${STEP}"

    echo ""
    echo "--- step${STEP} ---"
    echo "  Local: $LOCAL_PATH"
    echo "  Remote: $REPO_ID / $PATH_IN_REPO"

    if [ ! -d "$LOCAL_PATH" ]; then
        echo "  SKIP: local path does not exist"
        continue
    fi

    if ! ls "$LOCAL_PATH"/model-*.safetensors 1>/dev/null 2>&1; then
        echo "  SKIP: no safetensors found in $LOCAL_PATH"
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  DRY RUN: would upload $LOCAL_PATH -> $REPO_ID/$PATH_IN_REPO"
    else
        echo "  Uploading..."
        huggingface-cli upload "$REPO_ID" "$LOCAL_PATH" "$PATH_IN_REPO" --repo-type model
        echo "  Done: https://huggingface.co/$REPO_ID/tree/main/$PATH_IN_REPO"
    fi
done

echo ""
echo "=============================================="
echo "Upload complete!"
echo "=============================================="
