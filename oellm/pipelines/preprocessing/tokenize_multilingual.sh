#!/bin/bash
#SBATCH --job-name=tokenize-multilingual
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/preprocessing/logs/tokenize_%A_%a.log
#SBATCH --array=0-1

# Tokenize multiple experiments in parallel (one per array task)

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
source "$PROJECT_ROOT/.venv/bin/activate"

export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

EXPERIMENTS=(A2-80en A3-70en)
EXP="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "Tokenizing ${EXP}..."
python "$PROJECT_ROOT/scripts/data/convert_sft_data_for_olmocore.py" \
    --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct-SFT \
    --dataset_mixer_list "$PROJECT_ROOT/data/datasets_multilingual_sft/assembled/${EXP}.parquet" 1.0 \
    --output_dir "$PROJECT_ROOT/data/datasets_multilingual_sft/tokenized/${EXP}" \
    --chat_template_name olmo \
    --max_seq_length 32768

echo "Done tokenizing ${EXP}"
