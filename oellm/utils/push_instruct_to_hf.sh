#!/bin/bash
#SBATCH --job-name=push-instruct-hf
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/utils/logs/push_instruct_hf_%j.log

# Upload all instruct SFT checkpoints to HuggingFace
# Repo: openeurollm/OLMo-3-7B-Instruct-SFT

set -euo pipefail

VENV_PYTHON="/work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/python"
export HF_HOME="/work/dlclarge2/ferreira-oellm/open-instruct/models/huggingface"

HF_DIR="/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka-hf-all"
REPO_ID="openeurollm/OLMo-3-7B-Instruct-SFT"
MODEL_CARD="/work/dlclarge2/ferreira-oellm/open-instruct/oellm/utils/instruct_sft_model_card.md"

STEPS=(1000 2000 3000 3150 3252)

# Verify all checkpoints exist
for step in "${STEPS[@]}"; do
    if [ ! -d "$HF_DIR/step${step}" ]; then
        echo "Error: $HF_DIR/step${step} not found. Run convert_horeka_instruct_all.sh first."
        exit 1
    fi
done

echo "Uploading to $REPO_ID..."

$VENV_PYTHON -c "
from huggingface_hub import HfApi
import shutil, os

api = HfApi()
repo_id = '$REPO_ID'
hf_dir = '$HF_DIR'
model_card = '$MODEL_CARD'
steps = [1000, 2000, 3000, 3150, 3252]

# Upload README
print('Uploading README.md...')
api.upload_file(
    path_or_fileobj=model_card,
    path_in_repo='README.md',
    repo_id=repo_id,
    repo_type='model',
)

# Upload each checkpoint
for step in steps:
    step_dir = os.path.join(hf_dir, f'step{step}')
    print(f'Uploading step{step} from {step_dir}...')
    api.upload_folder(
        folder_path=step_dir,
        path_in_repo=f'step{step}',
        repo_id=repo_id,
        repo_type='model',
    )
    print(f'  step{step} uploaded!')

print()
print(f'All checkpoints uploaded to https://huggingface.co/{repo_id}')
"

echo "Done!"
