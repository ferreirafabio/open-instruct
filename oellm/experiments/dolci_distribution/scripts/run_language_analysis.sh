#!/bin/bash
#SBATCH --job-name=lang-dist
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --array=0-63
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/dolci_distribution/logs/lang_dist_%A_%a.log

set -euo pipefail

# --- Arguments ---
DATASET="${1:?Usage: sbatch run_language_analysis.sh <instruct|think>}"
if [[ "$DATASET" != "instruct" && "$DATASET" != "think" ]]; then
    echo "ERROR: Dataset must be 'instruct' or 'think', got '$DATASET'"
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

# --- Elastic sharding ---
# SLURM_ARRAY_TASK_COUNT is set when running as an array job.
# Fall back to NUM_SHARDS env var, then to 1 (single node).
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-${NUM_SHARDS:-1}}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

# --- Status ---
echo "============================================"
echo "Language Distribution Analysis"
echo "============================================"
echo "Dataset:    $DATASET"
echo "Shard:      $SHARD_INDEX / $NUM_SHARDS"
echo "Workers:    $SLURM_CPUS_PER_TASK"
echo "Output dir: $OUTPUT_DIR"
echo "Host:       $(hostname)"
echo "Start time: $(date)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

$VENV_PYTHON "$SCRIPT" detect \
    --dataset "$DATASET" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUTPUT_DIR"

echo "============================================"
echo "Shard $SHARD_INDEX finished at $(date)"
echo "============================================"
