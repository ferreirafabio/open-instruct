#!/usr/bin/env bash
#SBATCH --job-name=tokenize-E1-90en
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/tokenization/logs/tokenize_trackE_%A.log

# Tokenize Track E assembled mixture for OLMo-core training
#
# Track E: Dolci-Instruct-SFT English + EU multilingual sources (~500k samples)
# Same as Track D but scaled up to test data scaling with Dolci replay.
#
# Prerequisites: Run assemble_mixture.py first:
#   python oellm/pipelines/preprocessing/assemble_mixture.py --config oellm/configs/multilingual_trackE_90en.yaml
#
# Usage:
#   sbatch oellm/pipelines/tokenization/tokenize_trackE.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
INPUT_PARQUET="$PROJECT_ROOT/data/datasets_multilingual_sft/assembled/E1-90en.parquet"
OUTPUT_DIR="$PROJECT_ROOT/data/datasets_multilingual_sft/tokenized/E1-90en"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"
MAX_SEQ_LENGTH=32768

export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

WORKER_COUNT=$((${SLURM_CPUS_PER_TASK:-48} * 3 / 4))
export BEAKER_ASSIGNED_CPU_COUNT="$WORKER_COUNT"

source "$PROJECT_ROOT/.venv/bin/activate"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Track E Tokenization: E1-90en"
echo "=============================================="
echo "Input: $INPUT_PARQUET"
echo "Output: $OUTPUT_DIR"
echo "Tokenizer: $TOKENIZER"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "Workers: $WORKER_COUNT"
echo "=============================================="

if [ ! -f "$INPUT_PARQUET" ]; then
    echo "ERROR: Assembled parquet not found: $INPUT_PARQUET"
    exit 1
fi

python "$PROJECT_ROOT/scripts/data/convert_sft_data_for_olmocore.py" \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list "$INPUT_PARQUET" 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --chat_template_name olmo \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --visualize

echo ""
echo "=============================================="
echo "TOKENIZATION COMPLETE: E1-90en"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
