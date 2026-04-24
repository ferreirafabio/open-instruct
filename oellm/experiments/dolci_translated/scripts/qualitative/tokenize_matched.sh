#!/usr/bin/env bash
#SBATCH --job-name=tokenize-dt-matched
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/dolci_translated/logs/%j.tokenize-matched.out
#SBATCH --error=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/dolci_translated/logs/%j.tokenize-matched.err

# Tokenize the matched-compute A-25en assembled parquet for OLMo-core training.
#
# Usage:
#   sbatch oellm/experiments/dolci_translated/scripts/qualitative/tokenize_matched.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"
MAX_SEQ_LENGTH=32768
EXPERIMENT="dt-A-25en-matched"

INPUT_PARQUET="$PROJECT_ROOT/data/datasets_multilingual_sft/assembled/${EXPERIMENT}.parquet"
OUTPUT_DIR="$PROJECT_ROOT/data/datasets_multilingual_sft/tokenized/${EXPERIMENT}"

export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

WORKER_COUNT=$((${SLURM_CPUS_PER_TASK:-48} * 3 / 4))
export BEAKER_ASSIGNED_CPU_COUNT="$WORKER_COUNT"

source "$PROJECT_ROOT/.venv/bin/activate"
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$INPUT_PARQUET" ]; then
    echo "ERROR: Assembled parquet not found: $INPUT_PARQUET"
    exit 1
fi

echo "Tokenizing $EXPERIMENT (input: $INPUT_PARQUET)"

python "$PROJECT_ROOT/scripts/data/convert_sft_data_for_olmocore.py" \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list "$INPUT_PARQUET" 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --chat_template_name olmo \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --visualize

echo "TOKENIZATION COMPLETE: $EXPERIMENT  (output: $OUTPUT_DIR)"
