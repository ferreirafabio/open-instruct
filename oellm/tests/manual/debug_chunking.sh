#!/bin/bash
#SBATCH --job-name=debug-chunking
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/translation/logs/debug_chunking_%j.log

# Debug chunking: show individual chunks and merged output

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

source "$PROJECT_ROOT/.venv/bin/activate"

python oellm/tests/manual/debug_chunking.py
