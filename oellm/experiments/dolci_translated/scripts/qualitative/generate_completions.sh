#!/bin/bash
#SBATCH --job-name=gen-dt-qual
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/dolci_translated/logs/gen_qual_%j.log

# Generate qualitative completions for the dolci_translated webpage.
# Loads 12 models sequentially, vLLM batched generation, writes site/completions.json
# incrementally so requeue picks up where it left off.
#
# Usage:
#   sbatch oellm/experiments/dolci_translated/scripts/qualitative/generate_completions.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

cd "$PROJECT_ROOT"

$VENV_PYTHON oellm/experiments/dolci_translated/scripts/qualitative/generate_completions.py

# Auto-deploy refreshed completions to github.io + HF Space (skip with NO_DEPLOY=1).
# Runs the Playwright suite first; deploy aborts on failure.
if [ "${NO_DEPLOY:-0}" = "0" ]; then
    echo "==> Auto-deploying refreshed completions.json"
    bash "$PROJECT_ROOT/oellm/experiments/dolci_translated/scripts/qualitative/deploy.sh" || {
        echo "WARN: deploy failed; completions.json saved locally but NOT pushed"
        exit 0  # don't propagate deploy failure as gen failure
    }
fi
