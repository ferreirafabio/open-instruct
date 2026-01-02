#!/bin/bash
#SBATCH --job-name=download-olmo3-instruct-sft
#SBATCH --partition=bosch_cpu-cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/logs/download_baseline_%j.log

# Download the official OLMo-3-7B-Instruct-SFT baseline model from HuggingFace
# This is the SFT-only checkpoint (before DPO/RLVR), matching our training stage.
# See: https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT

set -e

# Use absolute paths (relative paths break on compute nodes)
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"

# Model to download
MODEL_ID="allenai/Olmo-3-7B-Instruct-SFT"

# Clean directory name (no nested HF cache structure)
TARGET_DIR="$PROJECT_ROOT/models/baselines/Olmo-3-7B-Instruct-SFT"

echo "=== Downloading $MODEL_ID ==="
echo "Target directory: $TARGET_DIR"
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
echo "Directory structure:"
echo "  models/baselines/"
echo "    Olmo-3-7B-Instruct-SFT/   <-- Downloaded (SFT baseline for comparison)"
echo "    Olmo-3-1025-7B/           # Base model (pre-training, HF format)"
echo "    Olmo-3-1025-7B-olmocore/  # Base model (OLMo-core format)"
echo ""
echo "You can now use this model for evaluation comparisons."

