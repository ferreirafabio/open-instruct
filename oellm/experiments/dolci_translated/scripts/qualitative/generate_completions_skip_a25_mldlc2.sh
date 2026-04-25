#!/bin/bash
#SBATCH --job-name=gen-dt-qual-mldlc2
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/dolci_translated/logs/gen_qual_mldlc2_%j.log

# Same as generate_completions_skip_a25.sh but submitted to mldlc2_gpu-h200
# to race the alldlc2_gpu-h200 queue. generate_completions.py loads state from
# site/completions.json and skips (model, prompt) pairs already done, so whichever
# partition starts first will finish first and the other will idle out quickly.
#
# Usage:
#   sbatch oellm/experiments/dolci_translated/scripts/qualitative/generate_completions_skip_a25_mldlc2.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

cd "$PROJECT_ROOT"

$VENV_PYTHON oellm/experiments/dolci_translated/scripts/qualitative/generate_completions.py \
    --skip-group A-25en

if [ "${NO_DEPLOY:-0}" = "0" ]; then
    echo "==> Auto-deploying refreshed completions.json"
    bash "$PROJECT_ROOT/oellm/experiments/dolci_translated/scripts/qualitative/deploy.sh" || {
        echo "WARN: deploy failed; completions.json saved locally but NOT pushed"
        exit 0
    }
fi
