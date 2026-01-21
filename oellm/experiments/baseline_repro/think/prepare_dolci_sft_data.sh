#!/usr/bin/env bash
#SBATCH --job-name=tokenize-think
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=6
#SBATCH --nodes=1-2
#SBATCH --time=1-00:00:00
#SBATCH --array=1-9%1
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/baseline_repro/think/logs/%A_%a.%x.out
#SBATCH --error=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/baseline_repro/think/logs/%A_%a.%x.err

# Converts allenai/Dolci-Think-SFT-7B for Olmo-core SFT training.
# Usage: sbatch prepare_dolci_sft_data.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/baseline_reproduction/dolci_think_sft_tokenized_v2}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Think-SFT}"

# Limit parallel workers to avoid OOM (each worker loads data into memory)
# 192 workers for 6 GPUs (~1125GB RAM, ~288 CPUs)
export BEAKER_ASSIGNED_CPU_COUNT=192

# Use project-level HuggingFace cache directories
export HF_HOME="${PROJECT_ROOT}/models/huggingface"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/data/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRITON_CACHE_DIR="${PROJECT_ROOT}/.cache/triton"
export HF_TOKEN="${HF_TOKEN:-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}"

echo "=== Tokenization Configuration ==="
echo "TOKENIZER=$TOKENIZER"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "=================================="

mkdir -p "$OUTPUT_DIR" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRITON_CACHE_DIR"

cd "$PROJECT_ROOT"
source .venv/bin/activate

# Start background RAM monitor (logs every 60s)
# Use SLURM-allocated memory, not total node memory
echo "=== SLURM Memory Info ==="
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-NOT_SET}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-NOT_SET}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NOT_SET}"

# Try SLURM_MEM_PER_NODE first, then scontrol
if [ -n "${SLURM_MEM_PER_NODE:-}" ] && [ "${SLURM_MEM_PER_NODE}" -gt 0 ] 2>/dev/null; then
  SLURM_MEM_GB=$((SLURM_MEM_PER_NODE / 1024))
  echo "Memory source: SLURM_MEM_PER_NODE (${SLURM_MEM_PER_NODE}MB = ${SLURM_MEM_GB}GB)"
else
  # Get from scontrol (parses "mem=1500G" or "mem=192000M")
  MEM_STR=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'mem=\K[0-9]+[GM]' | head -1)
  echo "scontrol mem string: ${MEM_STR:-NOT_FOUND}"
  if [ -n "$MEM_STR" ]; then
    MEM_VAL=$(echo "$MEM_STR" | grep -oP '[0-9]+')
    MEM_UNIT=$(echo "$MEM_STR" | grep -oP '[GM]')
    if [ "$MEM_UNIT" = "G" ]; then
      SLURM_MEM_GB=$MEM_VAL
    else
      SLURM_MEM_GB=$((MEM_VAL / 1024))
    fi
    echo "Memory source: scontrol (${MEM_STR} = ${SLURM_MEM_GB}GB)"
  else
    echo "ERROR: Cannot determine SLURM allocated memory!"
    exit 1
  fi
fi
echo "========================="
(
  while true; do
    MEM_USED=$(free -g | awk '/^Mem:/{print $3}')
    MEM_PCT=$((MEM_USED * 100 / SLURM_MEM_GB))
    echo "[RAM $(date '+%H:%M:%S')] ${MEM_USED}GB / ${SLURM_MEM_GB}GB (${MEM_PCT}%)"
    sleep 60
  done
) &
RAM_MONITOR_PID=$!
trap "kill $RAM_MONITOR_PID 2>/dev/null" EXIT

python scripts/data/convert_sft_data_for_olmocore.py \
  --tokenizer_name_or_path "$TOKENIZER" \
  --dataset_mixer_list allenai/Dolci-Think-SFT-7B 1.0 \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 32768 \
  --visualize True \
  --resume \
  --checkpoint_interval 100000

echo "=== Tokenization Complete ==="
echo "Output saved to: $OUTPUT_DIR"
