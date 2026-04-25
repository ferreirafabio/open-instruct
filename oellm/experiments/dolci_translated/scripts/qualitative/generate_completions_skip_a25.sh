#!/bin/bash
#SBATCH --job-name=gen-dt-qual-skip-a25
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/dolci_translated/logs/gen_qual_skip_a25_%j.log

# Generate qualitative completions for the new 7-language stratified prompt set,
# skipping the A-25en group (matched re-run is in progress; its completions will
# be filled in by the post-training cron).
#
# 11 models × ~698 prompts ≈ 7,700 generations on a single H200.
#
# Usage:
#   sbatch oellm/experiments/dolci_translated/scripts/qualitative/generate_completions_skip_a25.sh

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
