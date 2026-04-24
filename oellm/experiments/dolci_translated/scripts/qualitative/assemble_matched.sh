#!/usr/bin/env bash
#SBATCH --job-name=assemble-dt-matched
#SBATCH --partition=alldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/dolci_translated/logs/%j.assemble-matched.out
#SBATCH --error=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/dolci_translated/logs/%j.assemble-matched.err

# Assemble the matched-compute Dolci-Translated A-25en mixture (2.87M total,
# matched to A-75en).
#
# Usage:
#   sbatch oellm/experiments/dolci_translated/scripts/qualitative/assemble_matched.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

CONFIG="dolci_translated_A_25en_matched"

echo "Assembling ${CONFIG}..."
python "$PROJECT_ROOT/oellm/pipelines/preprocessing/assemble_mixture.py" \
    --config "$PROJECT_ROOT/oellm/configs/${CONFIG}.yaml" \
    --by-language-dir "$PROJECT_ROOT/data/datasets_multilingual_sft/by_language" \
    --output-dir "$PROJECT_ROOT/data/datasets_multilingual_sft/assembled"

echo "Done: ${CONFIG}"
