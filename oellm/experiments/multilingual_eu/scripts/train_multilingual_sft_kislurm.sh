#!/usr/bin/env bash
#SBATCH --job-name=sft-multilingual
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=12:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/multilingual_eu/logs/%A_%a.%x.%N.out
#SBATCH --error=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/multilingual_eu/logs/%A_%a.%x.%N.err
#SBATCH --array=0-9%1
#SBATCH --requeue

# Multilingual SFT training script for OLMo-core
# Continues training the instruct checkpoint on multilingual data.
#
# Configure via environment variables:
#   EXPERIMENT    - Experiment name (e.g., A1-90en, B1-90en)
#   DATASET_PATH  - Path to tokenized dataset
#   SAVE_INTERVAL - Ephemeral checkpoint save interval (default: 50 for Track A, 200 for Track B)
#
# Usage:
#   EXPERIMENT=A1-90en sbatch oellm/experiments/multilingual_eu/scripts/train_multilingual_sft_kislurm.sh
#   EXPERIMENT=B1-90en SAVE_INTERVAL=200 sbatch oellm/experiments/multilingual_eu/scripts/train_multilingual_sft_kislurm.sh
#   TEST_RUN=true EXPERIMENT=A1-90en sbatch oellm/experiments/multilingual_eu/scripts/train_multilingual_sft_kislurm.sh

set -euo pipefail

# Load env vars from .env if present
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
if [ -f "$PROJECT_ROOT/.env" ]; then
  export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

source "$PROJECT_ROOT/.venv/bin/activate"

# Ensure we use the OLMo-core source checkout (with local patches)
OLMOCORE_PATH="/work/dlclarge2/ferreira-oellm/OLMo-core"
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

# Experiment configuration
EXPERIMENT="${EXPERIMENT:?Usage: EXPERIMENT=A1-90en sbatch oellm/experiments/multilingual_eu/scripts/train_multilingual_sft_kislurm.sh}"
RUN_NAME="dolci-instruct-eu-${EXPERIMENT}"
CLUSTER_NAME="slurm"
GPUS="${GPUS:-8}"

# Dataset and checkpoint paths
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/datasets_multilingual_sft/tokenized/${EXPERIMENT}}"
BASE_CKPT="${BASE_CKPT:-${PROJECT_ROOT}/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/.cache}"

# Hyperparameters (from OLMo-3 paper Table 47)
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
SEQ_LEN="${SEQ_LEN:-32768}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((SEQ_LEN * 32))}"  # 1M tokens
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"  # 50 for Track A, 200 for Track B

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/models/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-nvlink}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Test run support
TEST_RUN="${TEST_RUN:-false}"
TEST_STEPS="${TEST_STEPS:-20}"
EXTRA_ACCELERATE_ARGS=()
if [[ "${TEST_RUN}" == "true" ]]; then
  echo "TEST_RUN=true: limiting this submission to ${TEST_STEPS} steps."
  EXTRA_ACCELERATE_ARGS+=(
    "--trainer.max_duration.value=${TEST_STEPS}"
    "--trainer.max_duration.unit=steps"
    "--trainer.callbacks.checkpointer.save_interval=${TEST_STEPS}"
    "--trainer.callbacks.checkpointer.ephemeral_save_interval=${TEST_STEPS}"
  )
fi

# W&B configuration
WANDB_ENABLED="${WANDB_ENABLED:-auto}"
if [[ "${WANDB_ENABLED}" == "auto" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ENABLED=true
  else
    WANDB_ENABLED=false
  fi
fi
WANDB_PROJECT="${WANDB_PROJECT:-olmo-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS_JSON="${WANDB_TAGS_JSON:-[\"multilingual\",\"eu\",\"instruct\",\"sft\",\"7b\",\"${EXPERIMENT}\"]}"
WANDB_CANCEL_CHECK_INTERVAL="${WANDB_CANCEL_CHECK_INTERVAL:-1000000000}"
WANDB_CANCEL_TAGS_JSON="${WANDB_CANCEL_TAGS_JSON:-[]}"

mkdir -p "$PROJECT_ROOT/oellm/experiments/multilingual_eu/logs" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_HUB_CACHE"

echo ""
echo "=============================================="
echo "MULTILINGUAL SFT TRAINING"
echo "=============================================="
echo "EXPERIMENT=$EXPERIMENT"
echo "RUN_NAME=$RUN_NAME"
echo "DATASET=$DATASET_PATH"
echo "BASE_CKPT=$BASE_CKPT"
echo "GPUS=$GPUS"
echo "SEQ_LEN=$SEQ_LEN"
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "SAVE_INTERVAL=$SAVE_INTERVAL"
echo "=============================================="
echo ""

NUM_GPUS=$GPUS
NUM_MACHINES=1
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
MAIN_PROCESS_PORT=29501

srun accelerate launch \
  --mixed_precision bf16 \
  --num_machines $NUM_MACHINES \
  --num_processes $NUM_GPUS \
  --machine_rank $MACHINE_RANK \
  --main_process_ip $MAIN_PROCESS_IP \
  --main_process_port $MAIN_PROCESS_PORT \
  --use_deepspeed \
  --deepspeed_config_file configs/ds_configs/stage2_accelerate.conf \
  --deepspeed_multinode_launcher standard \
  "${OLMOCORE_PATH}/src/scripts/train/sft/OLMo-sft.py" train \
    "$RUN_NAME" \
    "$BASE_CKPT" \
    "$CLUSTER_NAME" \
    --seq_len="$SEQ_LEN" \
    --num_nodes=$NUM_MACHINES \
    --global_batch_size="$GLOBAL_BATCH_SIZE" \
    --model_name="olmo3-7b" \
    --dataset_path="$DATASET_PATH" \
    --train_module.optim.lr="$LEARNING_RATE" \
    --trainer.max_duration.value=2 \
    --trainer.max_duration.unit=epochs \
    --trainer.callbacks.checkpointer.ephemeral_save_interval="$SAVE_INTERVAL" \
    --trainer.callbacks.wandb.enabled="$WANDB_ENABLED" \
    --trainer.callbacks.wandb.project="$WANDB_PROJECT" \
    --trainer.callbacks.wandb.entity="$WANDB_ENTITY" \
    --trainer.callbacks.wandb.name="$RUN_NAME" \
    --trainer.callbacks.wandb.tags="$WANDB_TAGS_JSON" \
    --trainer.callbacks.wandb.cancel_check_interval="$WANDB_CANCEL_CHECK_INTERVAL" \
    --trainer.callbacks.wandb.cancel_tags="$WANDB_CANCEL_TAGS_JSON" \
    --save_tokenizer=True \
    --budget=unused \
    --workspace=unused \
    "${EXTRA_ACCELERATE_ARGS[@]}"
