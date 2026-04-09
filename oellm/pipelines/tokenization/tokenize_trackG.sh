#!/usr/bin/env bash
#SBATCH --job-name=tokenize-G
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/tokenization/logs/tokenize_trackG_%A_%a.log

# Tokenize Track G assembled mixtures (166k matched, 4 EU langs)
# Usage: sbatch --array=0-4 oellm/pipelines/tokenization/tokenize_trackG.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"
MAX_SEQ_LENGTH=32768

EXPERIMENTS=("G1-100en" "G2-75en" "G3-50en" "G4-25en" "G5-0en")
TASK_ID="${SLURM_ARRAY_TASK_ID:?Must run as array job}"
EXPERIMENT="${EXPERIMENTS[$TASK_ID]}"

INPUT_PARQUET="$PROJECT_ROOT/data/datasets_multilingual_sft/assembled/${EXPERIMENT}.parquet"
OUTPUT_DIR="$PROJECT_ROOT/data/datasets_multilingual_sft/tokenized/${EXPERIMENT}"

export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
WORKER_COUNT=$((${SLURM_CPUS_PER_TASK:-48} * 3 / 4))
export BEAKER_ASSIGNED_CPU_COUNT="$WORKER_COUNT"

source "$PROJECT_ROOT/.venv/bin/activate"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Track G Tokenization: $EXPERIMENT"
echo "=============================================="
echo "Input: $INPUT_PARQUET"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

[ -f "$INPUT_PARQUET" ] || { echo "ERROR: $INPUT_PARQUET not found"; exit 1; }

python "$PROJECT_ROOT/scripts/data/convert_sft_data_for_olmocore.py" \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list "$INPUT_PARQUET" 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --chat_template_name olmo \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --visualize

echo "TOKENIZATION COMPLETE: $EXPERIMENT"
