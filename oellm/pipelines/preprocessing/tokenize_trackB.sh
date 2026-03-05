#!/bin/bash
#SBATCH --job-name=tokenize-trackB
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/preprocessing/logs/tokenize_%A_%a.log
#SBATCH --array=0-1

# Tokenize Track B experiments on CPU partition (no GPU needed for tokenization).
# Array 0 = B1-90en, Array 1 = B2-80en

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
source "$PROJECT_ROOT/.venv/bin/activate"

export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

EXPERIMENTS=(B1-90en B2-80en)
EXP="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "Tokenizing ${EXP}..."
python "$PROJECT_ROOT/scripts/data/convert_sft_data_for_olmocore.py" \
    --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct-SFT \
    --dataset_mixer_list "$PROJECT_ROOT/data/datasets_multilingual_sft/assembled/${EXP}.parquet" 1.0 \
    --output_dir "$PROJECT_ROOT/data/datasets_multilingual_sft/tokenized/${EXP}" \
    --chat_template_name olmo \
    --max_seq_length 32768

echo "Done tokenizing ${EXP}"
