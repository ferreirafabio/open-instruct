#!/bin/bash
#SBATCH --job-name=split-by-language
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --array=0-3
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/preprocessing/logs/split_%A_%a.log

# Split preprocessed datasets by language — one SLURM array task per dataset (4 nodes in parallel)
# Run after preprocess_multilingual_slurm.sh completes.
#
# Usage:
#   sbatch --dependency=afterok:$PREPROCESS_JOB_ID oellm/pipelines/preprocessing/split_multilingual_slurm.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

DATASETS=(
    "CohereLabs-fusion-synth-data-ufb"
    "allenai-WildChat-1M"
    "lmsys-lmsys-chat-1m-multilingual"
    "OpenAssistant-oasst2"
)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

echo "=============================================="
echo "Splitting by language: $DATASET"
echo "Task: ${SLURM_ARRAY_TASK_ID}/${#DATASETS[@]}"
echo "=============================================="

$VENV_PYTHON "$PROJECT_ROOT/oellm/pipelines/preprocessing/split_by_language.py" \
    --input-dir "$PROJECT_ROOT/data/datasets_multilingual_sft/preprocessed" \
    --output-dir "$PROJECT_ROOT/data/datasets_multilingual_sft/by_language" \
    --datasets "$DATASET"

echo "Done splitting: $DATASET"
