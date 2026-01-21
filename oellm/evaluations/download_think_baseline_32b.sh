#!/bin/bash
#SBATCH --job-name=download-olmo3-think-sft-32b
#SBATCH --partition=bosch_cpu-cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/logs/download_think_baseline_32b_%j.log

# Download the OLMo-3-32B-Think-SFT baseline model from HuggingFace
# See: https://huggingface.co/allenai/Olmo-3-32B-Think-SFT

set -euo pipefail

# Use absolute paths (relative paths break on compute nodes)
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"

# Model to download
MODEL_ID="allenai/Olmo-3-32B-Think-SFT"

# Clean directory name (no nested HF cache structure)
TARGET_DIR="$PROJECT_ROOT/models/baselines/Olmo-3-32B-Think-SFT"

# Use project cache instead of home directory
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"

echo "=== Downloading $MODEL_ID ==="
echo "Target directory: $TARGET_DIR"
echo "HF_HOME: $HF_HOME"
echo ""

# Activate the virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Download the model using huggingface_hub
export TARGET_DIR
export MODEL_ID
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

model_id = os.environ["MODEL_ID"]
target_dir = os.environ["TARGET_DIR"]

print(f"Downloading {model_id} to: {target_dir}")
snapshot_download(
    model_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False,
)
print("Download complete!")
EOF

echo ""
echo "=== Download Complete ==="
echo "Model saved to: $TARGET_DIR"
echo ""
echo "Next steps:"
echo "  1) Run think evaluation with the 32B baseline (override BASELINE path):"
echo "     BASELINE=$TARGET_DIR \\"
echo "     TRAINED=$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf \\"
echo "     sbatch oellm/evaluations/benchmarks/run_evaluation.sh think all"
echo ""

