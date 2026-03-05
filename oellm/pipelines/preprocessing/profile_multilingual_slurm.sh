#!/bin/bash
#SBATCH --job-name=profile-multilingual
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --array=0-3
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/preprocessing/logs/profile_%A_%a.log

# Profile multilingual datasets — one SLURM array task per dataset (4 nodes in parallel)
# Each task downloads and profiles one dataset, saving per-dataset JSON.
#
# After all tasks complete, merge:
#   sbatch --dependency=afterok:$JOB_ID oellm/pipelines/preprocessing/merge_profiles.sh
#
# Usage:
#   JOB_ID=$(sbatch --parsable oellm/pipelines/preprocessing/profile_multilingual_slurm.sh)
#   sbatch --dependency=afterok:$JOB_ID oellm/pipelines/preprocessing/merge_profiles.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

DATASETS=(fusion-synth wildchat lmsys-chat oasst2)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

echo "=============================================="
echo "Profiling dataset: $DATASET"
echo "Task: ${SLURM_ARRAY_TASK_ID}/${#DATASETS[@]}"
echo "=============================================="

mkdir -p "$PROJECT_ROOT/oellm/pipelines/preprocessing/logs"

$VENV_PYTHON "$PROJECT_ROOT/oellm/pipelines/preprocessing/profile_multilingual_datasets.py" \
    --datasets "$DATASET" \
    --output "$PROJECT_ROOT/data/datasets_multilingual_sft/profiles"

echo "Done profiling: $DATASET"
