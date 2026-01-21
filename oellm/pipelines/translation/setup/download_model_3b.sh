#!/bin/bash
#SBATCH --job-name=download-nllb
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/translation/logs/download_%j.log

# Download NLLB-200-3.3B translation model
#
# This downloads the model to the HuggingFace cache so it's available
# for translation jobs. Run this ONCE before starting translations.
#
# Usage:
#   sbatch oellm/pipelines/translation/download_model.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
cd "$PROJECT_ROOT"

# Store HuggingFace models in open-instruct, not home directory
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

mkdir -p oellm/pipelines/translation/logs

echo "HF_HOME: $HF_HOME"

echo "=========================================="
echo "DOWNLOADING NLLB-200-3.3B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "=========================================="

# Download the model using transformers
uv run python -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = 'facebook/nllb-200-3.3B'

print(f'Downloading tokenizer: {model_name}')
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f'Tokenizer downloaded. Vocab size: {tokenizer.vocab_size}')

print(f'Downloading model: {model_name}')
print('This may take 10-20 minutes...')
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)
print(f'Model downloaded. Parameters: {model.num_parameters():,}')

# Test that it works
print('Testing model...')
inputs = tokenizer('Hello, how are you?', return_tensors='pt')
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids('deu_Latn'), max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Test translation (en->de): {result}')

print()
print('Model downloaded and cached successfully!')
print('You can now run: sbatch oellm/pipelines/translation/translate_slurm.sh')
"

echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE"
echo "=========================================="
