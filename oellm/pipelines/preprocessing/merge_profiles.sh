#!/bin/bash
#SBATCH --job-name=merge-profiles
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:05:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/preprocessing/logs/merge_profiles_%j.log

# Merge per-dataset profiles into a single JSON
# Run after profile_multilingual_slurm.sh completes
#
# Usage:
#   sbatch --dependency=afterok:$PROFILE_JOB_ID oellm/pipelines/preprocessing/merge_profiles.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

$VENV_PYTHON "$PROJECT_ROOT/oellm/pipelines/preprocessing/profile_multilingual_datasets.py" \
    --merge \
    --output "$PROJECT_ROOT/data/datasets_multilingual_sft/profiles"
