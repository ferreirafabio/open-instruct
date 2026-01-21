#!/usr/bin/env bash
#SBATCH --job-name=download-sft-datasets
##SBATCH --partition=bosch_cpu-cascadelake
#SBATCH --partition=alldlc2_gpu-h200
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/preprocessing/logs/download_%j.log

# Download and preprocess SFT datasets from HuggingFace Hub
#
# This script downloads datasets and converts them to standard messages format.
# Each dataset is saved as a separate parquet file for flexible selection.
#
# Usage:
#   # Download all 19 datasets
#   sbatch download_datasets.sh
#
#   # Download specific datasets only
#   MIXTURES="nvidia-Nemotron mlabonne-open-perfectblend" sbatch download_datasets.sh

set -euo pipefail

# Load env vars from .env if present (for PROJECT_ROOT, proxies, etc.)
if [ -f /work/dlclarge2/ferreira-oellm/open-instruct/.env ]; then
    export $(grep -v '^#' /work/dlclarge2/ferreira-oellm/open-instruct/.env | xargs)
fi

# Configuration (PROJECT_ROOT comes from .env)
PROJECT_ROOT="${PROJECT_ROOT:-/work/dlclarge2/ferreira-oellm/open-instruct}"
SCRIPT_DIR="$PROJECT_ROOT/oellm/pipelines/preprocessing"

# Paths
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/datasets_mixture_sft_preprocessed}"

# Processing options
NUM_PROC="${NUM_PROC:-16}"

# Optional: specific mixtures to process (space-separated)
MIXTURES="${MIXTURES:-}"

# HuggingFace config
export HF_TOKEN="${HF_TOKEN:-}"  # Loaded from .env for gated datasets
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# Create directories
mkdir -p "$OUTPUT_DIR" "$SCRIPT_DIR/logs"

# Activate environment
source "$PROJECT_ROOT/.venv/bin/activate"

echo "=============================================="
echo "SFT Dataset Download & Preprocessing"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Num processes: $NUM_PROC"
echo "=============================================="
echo ""

# Build arguments
ARGS=(
    "--output-dir" "$OUTPUT_DIR"
    "--num-proc" "$NUM_PROC"
)

if [ -n "$MIXTURES" ]; then
    ARGS+=("--mixtures" $MIXTURES)
fi

# Run preprocessing
python "$SCRIPT_DIR/preprocess_datasets.py" "${ARGS[@]}"

echo ""
echo "=============================================="
echo "DOWNLOAD COMPLETE!"
echo "=============================================="
echo ""
echo "Datasets saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  For baseline reproduction (100% English):"
echo "    sbatch oellm/pipelines/tokenization/tokenize_baseline.sh"
echo ""
echo "  For EU24 experiments (with language mixing):"
echo "    1. Translate: sbatch oellm/pipelines/translation/translate_slurm.sh"
echo "    2. Tokenize:  CONFIG=mixture_eu24_90en_10eu.yaml sbatch oellm/pipelines/tokenization/tokenize_eu24.sh"
