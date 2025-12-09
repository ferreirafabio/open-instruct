#!/usr/bin/env bash
set -euo pipefail

# Converts allenai/Dolci-Think-SFT-32B for Olmo-core SFT training.
# Usage: ./prepare_dolci_sft_data.sh [OUTPUT_DIR] [TOKENIZER]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/dolci_think_sft_data}"
TOKENIZER="${2:-allenai/Olmo-3-1025-7B}"

CACHE_DIR="${OUTPUT_DIR}/hf_cache"
HF_HOME="${HF_HOME:-${CACHE_DIR}}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HOME
export HF_DATASETS_CACHE
export HF_MODULES_CACHE
export HF_HUB_CACHE

echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_MODULES_CACHE=$HF_MODULES_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"

mkdir -p "$OUTPUT_DIR" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_HUB_CACHE"

cd "$SCRIPT_DIR"

python scripts/data/convert_sft_data_for_olmocore.py \
  --tokenizer_name_or_path "$TOKENIZER" \
  --dataset_mixer_list allenai/Dolci-Think-SFT-7B 1.0 \
  --output_dir "$OUTPUT_DIR" \
  --chat_template_name olmo \
  --max_seq_length 32768 \
  --visualize True