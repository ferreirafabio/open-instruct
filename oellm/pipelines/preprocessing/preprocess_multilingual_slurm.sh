#!/bin/bash
#SBATCH --job-name=preprocess-multilingual
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/preprocessing/logs/preprocess_%A_%a.log

# Preprocess multilingual datasets — one SLURM array task per dataset (4 nodes in parallel)
# Downloads, transforms, and saves each dataset as parquet with messages + language columns.
#
# Usage:
#   sbatch oellm/pipelines/preprocessing/preprocess_multilingual_slurm.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

MIXTURES=(
    "CohereLabs-fusion-synth-data-ufb"
    "allenai-WildChat-1M"
    "lmsys-lmsys-chat-1m-multilingual"
    "OpenAssistant-oasst2"
)
MIXTURE="${MIXTURES[$SLURM_ARRAY_TASK_ID]}"

echo "=============================================="
echo "Preprocessing: $MIXTURE"
echo "Task: ${SLURM_ARRAY_TASK_ID}/${#MIXTURES[@]}"
echo "=============================================="

$VENV_PYTHON "$PROJECT_ROOT/oellm/pipelines/preprocessing/preprocess_datasets.py" \
    --mixtures "$MIXTURE" \
    --output-dir "$PROJECT_ROOT/data/datasets_multilingual_sft/preprocessed"

echo "Done preprocessing: $MIXTURE"
