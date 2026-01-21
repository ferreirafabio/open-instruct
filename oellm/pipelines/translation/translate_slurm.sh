#!/bin/bash
#SBATCH --job-name=translate-eu
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/translation/logs/%A_%a.log
#SBATCH --array=0-23%5
#SBATCH --requeue

# H200 nodes: 8 GPUs, 1500GB RAM, 384 CPUs per node
# Requesting 1 GPU gives us ~187GB RAM and ~48 CPUs proportionally
# NOTE: H200 partition has 24h max. For large datasets, the job will checkpoint
# progress and can be resumed by re-running. Each dataset checkpoints every 5000 examples.

# Unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# Translate SFT datasets to EU languages using NLLB-200-3.3B
#
# This script runs as a SLURM array job, with each task handling one language.
# Array indices 0-23 map to 24 EU languages (including English for cross-product translation).
#
# Usage:
#   # Translate all datasets to all languages (4 parallel jobs)
#   sbatch translate_slurm.sh
#
#   # Translate specific dataset only
#   DATASET=Anthropic-hh-rlhf.parquet sbatch translate_slurm.sh
#
#   # Translate to specific language only (array index)
#   sbatch --array=5 translate_slurm.sh  # German (de)
#
#   # Override parallelism
#   sbatch --array=0-23%8 translate_slurm.sh  # 8 parallel jobs
#
#   # Sample 10% of each dataset (for faster iteration)
#   SAMPLE=0.1 sbatch translate_slurm.sh
#
#   # Use smaller 600M model (faster but lower quality)
#   MODEL=facebook/nllb-200-distilled-600M sbatch translate_slurm.sh

set -euo pipefail

# Base paths
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"

# Store HuggingFace models in open-instruct, not home directory
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"
SCRIPT_DIR="${PROJECT_ROOT}/oellm/pipelines/translation"
INPUT_DIR="${PROJECT_ROOT}/data/datasets_mixture_sft_preprocessed"

# Output directory is determined by model and sample (set below after parsing env vars)

# Create log directory
mkdir -p "${SCRIPT_DIR}/logs"

# EU languages (24 total: 23 non-English + English for cross-product translation)
# Order matches array indices 0-23
LANGUAGES=(
    "bg"  # 0  - Bulgarian
    "hr"  # 1  - Croatian
    "cs"  # 2  - Czech
    "da"  # 3  - Danish
    "nl"  # 4  - Dutch
    "et"  # 5  - Estonian
    "fi"  # 6  - Finnish
    "fr"  # 7  - French
    "de"  # 8  - German
    "el"  # 9  - Greek
    "hu"  # 10 - Hungarian
    "ga"  # 11 - Irish
    "it"  # 12 - Italian
    "lv"  # 13 - Latvian
    "lt"  # 14 - Lithuanian
    "mt"  # 15 - Maltese
    "pl"  # 16 - Polish
    "pt"  # 17 - Portuguese
    "ro"  # 18 - Romanian
    "sk"  # 19 - Slovak
    "sl"  # 20 - Slovenian
    "es"  # 21 - Spanish
    "sv"  # 22 - Swedish
    "en"  # 23 - English (translates non-English rows to English)
)

# Get target language from array index
TARGET_LANG="${LANGUAGES[$SLURM_ARRAY_TASK_ID]}"

# Sample fraction (optional, default: no sampling)
SAMPLE_FRACTION="${SAMPLE:-}"
SAMPLE_ARG=""
SAMPLE_PCT="100"
if [ -n "$SAMPLE_FRACTION" ]; then
    SAMPLE_ARG="--sample-fraction $SAMPLE_FRACTION"
    # Convert fraction to percentage for dir name (0.01 -> 1, 0.1 -> 10)
    SAMPLE_PCT=$(echo "$SAMPLE_FRACTION * 100" | bc | cut -d. -f1)
fi

# Model selection (optional, default: 600M for speed)
MODEL_NAME="${MODEL:-facebook/nllb-200-distilled-600M}"
MODEL_ARG="--model $MODEL_NAME"

# Batch size for GPU inference (600M model uses ~2GB VRAM, H200 has 80GB)
# Trade-off: lower batch_size allows higher conversation_batch_size
BATCH_SIZE="${BATCH_SIZE:-256}"

# Checkpoint interval (save progress every N examples)
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"

# Conversations per batch (higher = faster throughput, more memory)
# 256 with BATCH_SIZE=256 should fit H200 80GB with 600M model
CONVERSATION_BATCH_SIZE="${CONVERSATION_BATCH_SIZE:-256}"

# Determine model short name for directory
if [[ "$MODEL_NAME" == *"600M"* ]]; then
    MODEL_SHORT="600m"
else
    MODEL_SHORT="3b"
fi

# Set output directory based on model and sample
OUTPUT_DIR="${PROJECT_ROOT}/data/datasets_mixture_sft_translated/eu24_${MODEL_SHORT}_${SAMPLE_PCT}pct"

echo "=========================================="
echo "TRANSLATION JOB"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Task:   $SLURM_ARRAY_TASK_ID"
echo "Target Lang:  $TARGET_LANG"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Model:        $MODEL_NAME"
echo "Batch Size:   $BATCH_SIZE (GPU inference)"
echo "Conv. Batch:  $CONVERSATION_BATCH_SIZE conversations/step"
echo "Checkpoint:   every $CHECKPOINT_INTERVAL examples"
echo "Output:       $OUTPUT_DIR"
if [ -n "$SAMPLE_FRACTION" ]; then
    echo "Sampling:     ${SAMPLE_FRACTION} (${SAMPLE_PCT}% of each dataset)"
else
    echo "Sampling:     100% (full datasets)"
fi
echo "=========================================="

# Datasets to translate (English -> target language)
# Note: lmsys-lmsys-chat-1m is ~34% non-English but we still translate since majority is English
DATASETS=(
    "nvidia-Nemotron-Post-Training-Dataset-v2.parquet"
    "HuggingFaceTB-smoltalk2.parquet"
    "mlabonne-open-perfectblend.parquet"
    "lmsys-lmsys-chat-1m.parquet"
    "Anthropic-hh-rlhf.parquet"
    "stanfordnlp-SHP.parquet"
    "berkeley-nest-Nectar.parquet"
    "lmarena-ai-arena-human-preference-55k.parquet"
    "argilla-ultrafeedback-binarized-preferences-cleaned.parquet"
    "nvidia-HelpSteer2.parquet"
    "nvidia-Aegis-AI-Content-Safety-Dataset-2.0.parquet"
    "argilla-distilabel-intel-orca-dpo-pairs.parquet"
    "HumanLLMs-Human-Like-DPO-Dataset.parquet"
    "argilla-distilabel-capybara-dpo-7k-binarized.parquet"
    "lmsys-mt_bench_human_judgments.parquet"
    "argilla-distilabel-math-preference-dpo.parquet"
    "jondurbin-truthy-dpo-v0.1.parquet"
)

# EXCLUDED:
# - microsoft-orca-agentinstruct-1M-v1.parquet (code-heavy, not conversational)
# - ministere-culture-comparia-votes.parquet (French dataset, not English source)

# If DATASET env var is set, translate only that one
if [ -n "${DATASET:-}" ]; then
    DATASETS=("$DATASET")
    echo "Processing single dataset: $DATASET"
fi

# Create output directory for this language
mkdir -p "${OUTPUT_DIR}/${TARGET_LANG}"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT"

# Translate each dataset
cd "$PROJECT_ROOT"
for DATASET in "${DATASETS[@]}"; do
    INPUT_FILE="${INPUT_DIR}/${DATASET}"
    OUTPUT_FILE="${OUTPUT_DIR}/${TARGET_LANG}/${DATASET}"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "Skipping (not found): $INPUT_FILE"
        continue
    fi

    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping (already exists): $OUTPUT_FILE"
        continue
    fi

    echo ""
    echo "Translating: $DATASET -> $TARGET_LANG"
    echo "Input:  $INPUT_FILE"
    echo "Output: $OUTPUT_FILE"

    # shellcheck disable=SC2086
    python oellm/pipelines/translation/translate_dataset.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --target-lang "$TARGET_LANG" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        --conversation-batch-size "$CONVERSATION_BATCH_SIZE" \
        --batch-size "$BATCH_SIZE" \
        $MODEL_ARG \
        $SAMPLE_ARG

    echo "Completed: $DATASET -> $TARGET_LANG"
done

echo ""
echo "=========================================="
echo "TRANSLATION COMPLETE: $TARGET_LANG"
echo "=========================================="
