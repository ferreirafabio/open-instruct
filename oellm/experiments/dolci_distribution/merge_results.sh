#!/bin/bash
#SBATCH --job-name=lang-dist-merge
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/dolci_distribution/logs/lang_dist_merge_%j.log

# Usage:
#   sbatch merge_results.sh <instruct|think|both>
#   sbatch --dependency=afterok:$ARRAY_JOB_ID merge_results.sh both

set -euo pipefail

# --- Arguments ---
DATASET="${1:?Usage: sbatch merge_results.sh <instruct|think|both>}"
if [[ "$DATASET" != "instruct" && "$DATASET" != "think" && "$DATASET" != "both" ]]; then
    echo "ERROR: Dataset must be 'instruct', 'think', or 'both', got '$DATASET'"
    exit 1
fi

# --- Paths (absolute, since SLURM copies scripts to spool) ---
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
SCRIPT="$PROJECT_ROOT/oellm/experiments/dolci_distribution/detect_language_distribution.py"
OUTPUT_DIR="$PROJECT_ROOT/oellm/experiments/dolci_distribution/results"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# --- HuggingFace cache ---
export HF_HOME="/work/dlclarge2/ferreira-oellm/open-instruct/models/huggingface"
export HF_DATASETS_CACHE="/work/dlclarge2/ferreira-oellm/open-instruct/data/huggingface"

# --- Status ---
echo "============================================"
echo "Merging Language Distribution Results"
echo "============================================"
echo "Dataset:    $DATASET"
echo "Output dir: $OUTPUT_DIR"
echo "Host:       $(hostname)"
echo "Start time: $(date)"
echo "============================================"

$VENV_PYTHON "$SCRIPT" merge \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR"

echo "============================================"
echo "Merge finished at $(date)"
echo "============================================"
