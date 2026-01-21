#!/usr/bin/env bash
#SBATCH --job-name=tokenize-think-h200
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --cpus-per-task=384
#SBATCH --mem=1400G
#SBATCH --gres=gpu:0
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/baseline_repro/think/logs/%j.%x.out
#SBATCH --error=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/baseline_repro/think/logs/%j.%x.err

# Converts allenai/Dolci-Think-SFT-7B for Olmo-core SFT training.
# Uses streaming writes to avoid OOM on large datasets.
# Usage: sbatch prepare_dolci_sft_data_h200.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/baseline_reproduction/dolci_think_sft_tokenized_v2}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Think-SFT}"

# Use all available CPUs for parallel dataset processing
export BEAKER_ASSIGNED_CPU_COUNT=384

# Use project-level HuggingFace cache directories
export HF_HOME="${PROJECT_ROOT}/models/huggingface"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/data/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRITON_CACHE_DIR="${PROJECT_ROOT}/.cache/triton"
export HF_TOKEN="${HF_TOKEN:-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}"

echo "=== Tokenization Configuration ==="
echo "TOKENIZER=$TOKENIZER"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "=================================="

mkdir -p "$OUTPUT_DIR" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRITON_CACHE_DIR"

cd "$PROJECT_ROOT"
source .venv/bin/activate

python scripts/data/convert_sft_data_for_olmocore.py \
  --tokenizer_name_or_path "$TOKENIZER" \
  --dataset_mixer_list allenai/Dolci-Think-SFT-7B 1.0 \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 32768 \
  --max_chunk_size_mb 1024 \
  --shuffle_seed 42 \
  --resume True \
  --checkpoint_interval 50000 \
  --visualize True

echo "=== Tokenization Complete ==="
echo "Output saved to: $OUTPUT_DIR"
