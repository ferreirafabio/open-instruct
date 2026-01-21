#!/usr/bin/env bash
#SBATCH --job-name=tokenize-eu24
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/tokenization/logs/tokenize_eu24_%j.log

# Tokenize EU24 translated datasets with language mixing
#
# This script tokenizes translated datasets using configurable language ratios.
# Uses the LanguageMixer utility for flexible per-language control.
#
# Usage:
#   # 90% English, 10% EU languages
#   CONFIG=mixture_eu24_90en_10eu.yaml sbatch tokenize_eu24.sh
#
#   # 80% English, 20% EU languages
#   CONFIG=mixture_eu24_80en_20eu.yaml sbatch tokenize_eu24.sh
#
#   # 100% English baseline
#   CONFIG=mixture_eu24_100en.yaml sbatch tokenize_eu24.sh
#
#   # Test with 1% sample
#   SAMPLE=0.01 CONFIG=mixture_eu24_90en_10eu.yaml sbatch tokenize_eu24.sh

set -euo pipefail

# Load env vars from .env if present
if [ -f /work/dlclarge2/ferreira-oellm/open-instruct/.env ]; then
    export $(grep -v '^#' /work/dlclarge2/ferreira-oellm/open-instruct/.env | xargs)
fi

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:-/work/dlclarge2/ferreira-oellm/open-instruct}"
SCRIPT_DIR="$PROJECT_ROOT/oellm/pipelines/tokenization"
CONFIG_DIR="$PROJECT_ROOT/oellm/configs"

# Config file (required)
CONFIG="${CONFIG:-mixture_eu24_90en_10eu.yaml}"
CONFIG_PATH="$CONFIG_DIR/$CONFIG"

# Paths
TRANSLATED_DIR="${TRANSLATED_DIR:-$PROJECT_ROOT/data/datasets_eu24_translated/600m_10pct}"

# Derive output directory from config file name
CONFIG_BASENAME=$(basename "$CONFIG" .yaml)
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/datasets_eu24_tokenized/${CONFIG_BASENAME#mixture_}}"

# Tokenizer
TOKENIZER="${TOKENIZER:-allenai/OLMo-2-1124-7B-Instruct}"

# Processing options
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32768}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000000}"
SAMPLE="${SAMPLE:-}"  # Optional sampling fraction for testing

# HuggingFace cache
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# Create directories
mkdir -p "$OUTPUT_DIR" "$SCRIPT_DIR/logs"

# Activate environment
source "$PROJECT_ROOT/.venv/bin/activate"

echo "=============================================="
echo "EU24 Dataset Tokenization with Language Mixing"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Config: $CONFIG_PATH"
echo "Translated dir: $TRANSLATED_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Tokenizer: $TOKENIZER"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "Total samples: $TOTAL_SAMPLES"
if [ -n "$SAMPLE" ]; then
    echo "Sample fraction: $SAMPLE (test mode)"
fi
echo "=============================================="
echo ""

# Check config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    echo ""
    echo "Available configs:"
    ls -1 "$CONFIG_DIR"/mixture_eu24_*.yaml 2>/dev/null || echo "  No configs found"
    exit 1
fi

# Build arguments
ARGS=(
    "--config" "$CONFIG_PATH"
    "--translated-dir" "$TRANSLATED_DIR"
    "--output-dir" "$OUTPUT_DIR"
    "--tokenizer" "$TOKENIZER"
    "--max-seq-length" "$MAX_SEQ_LENGTH"
    "--total-samples" "$TOTAL_SAMPLES"
)

if [ -n "$SAMPLE" ]; then
    ARGS+=("--sample" "$SAMPLE")
fi

# Run tokenization
python "$SCRIPT_DIR/tokenize_with_mixing.py" "${ARGS[@]}"

echo ""
echo "=============================================="
echo "TOKENIZATION COMPLETE!"
echo "=============================================="
echo ""
echo "Tokenized data saved to: $OUTPUT_DIR"
echo ""
echo "To train with OLMo-core:"
echo "  DATASET_PATH=$OUTPUT_DIR sbatch oellm/train/train_think_sft_dolci_7b_slurm.sh"
