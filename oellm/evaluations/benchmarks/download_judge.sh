#!/bin/bash
#SBATCH --job-name=download-judge
#SBATCH --partition=bosch_cpu-cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/logs/download_judge_%j.log

# Download the Qwen3-30B judge model for LLM evaluation

set -e

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
source "$PROJECT_ROOT/.venv/bin/activate"

# Use project cache instead of home directory
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"

echo "=== Downloading Judge Model ==="
echo "Model: $MODEL_ID"
echo ""

python3 -c "
from huggingface_hub import snapshot_download
print('Downloading $MODEL_ID...')
snapshot_download('$MODEL_ID')
print('Download complete!')
"

echo ""
echo "=== Download Complete ==="

