#!/usr/bin/env bash
#SBATCH --job-name=olmo3-7b-think-sft
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=6:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/train/logs/%A_%a.%x.%N.out
#SBATCH --error=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/train/logs/%A_%a.%x.%N.err
#SBATCH --array=0-9%1

# Generic SFT training script for OLMo-core
# Configure via environment variables:
#   DATASET_PATH - Path to tokenized dataset
#   BASE_CKPT    - Base checkpoint to finetune
#   RUN_NAME     - Experiment name
#
# Usage:
#   sbatch oellm/train/train_think_sft_dolci_7b_slurm.sh
#   DATASET_PATH=data/my_data RUN_NAME=my-exp sbatch oellm/train/train_think_sft_dolci_7b_slurm.sh

set -euo pipefail

# Load env vars from .env if present
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
if [ -f "$PROJECT_ROOT/.env" ]; then
  export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

source "$PROJECT_ROOT/.venv/bin/activate"

# Ensure we use the OLMo-core source checkout (with local patches) instead of any pip-installed version.
OLMOCORE_PATH="/work/dlclarge2/ferreira-oellm/OLMo-core"
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-dolci-think-sft}"
CLUSTER_NAME="slurm"
GPUS="${GPUS:-8}"
DATASET_PATH="${DATASET_PATH:-/work/dlclarge2/ferreira-oellm/open-instruct/data/baseline_reproduction/dolci_think_sft_tokenized_v2}"
BASE_CKPT="${BASE_CKPT:-/work/dlclarge2/ferreira-oellm/open-instruct/models/baselines/Olmo-3-1025-7B-olmocore}"
CACHE_DIR="${CACHE_DIR:-/work/dlclarge2/ferreira-oellm/open-instruct/.cache}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
SEQ_LEN="${SEQ_LEN:-32768}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((SEQ_LEN * 32))}" # 1M tokens (per baseline paper)
MAX_RANK_MICROBATCH_SIZE_TOKENS="${MAX_RANK_MICROBATCH_SIZE_TOKENS:-24576}" # 24K default, increase for fewer grad accum steps

export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-nvlink}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# Note: Don't set NCCL_ALGO=TREE as it doesn't support Broadcast with ncclInt8
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Support a short smoke test run while keeping the production hyperparameters unchanged.
TEST_RUN="${TEST_RUN:-false}"
TEST_STEPS="${TEST_STEPS:-20}"
EXTRA_ACCELERATE_ARGS=()
if [[ "${TEST_RUN}" == "true" ]]; then
  echo "TEST_RUN=true: limiting this submission to ${TEST_STEPS} steps."
  # Ephemeral save must be less than save_interval; use half the steps
  EPHEMERAL_INTERVAL=$((TEST_STEPS / 2))
  EXTRA_ACCELERATE_ARGS+=(
    "--trainer.max_duration.value=${TEST_STEPS}"
    "--trainer.max_duration.unit=steps"
    "--trainer.callbacks.checkpointer.save_interval=${TEST_STEPS}"
    "--trainer.callbacks.checkpointer.ephemeral_save_interval=${EPHEMERAL_INTERVAL}"
  )
fi

# W&B (OLMo-core uses WandBCallback). We keep this "auto" so jobs don't crash if WANDB_API_KEY isn't set.
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
# Auto-set WANDB_RUN_ID from RUN_NAME so resumed jobs continue the same W&B run
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}"
# Tags must be a *list* for the OLMo-core config parser. Use a JSON-ish list string.
WANDB_TAGS_JSON="${WANDB_TAGS_JSON:-[\"dolci\",\"think\",\"sft\",\"7b\"]}"
# W&B "cancel by tag" checks require W&B API connectivity; disable by default to avoid 30s stalls on clusters w/o egress.
WANDB_CANCEL_CHECK_INTERVAL="${WANDB_CANCEL_CHECK_INTERVAL:-1000000000}"
WANDB_CANCEL_TAGS_JSON="${WANDB_CANCEL_TAGS_JSON:-[]}"

mkdir -p "$PROJECT_ROOT/oellm/train/logs" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_HUB_CACHE"

echo "RUN_NAME=$RUN_NAME"
echo "DATASET=$DATASET_PATH"
echo "BASE_CKPT=$BASE_CKPT"
echo "GPUS=$GPUS"
echo "SEQ_LEN=$SEQ_LEN"
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "MAX_RANK_MICROBATCH_SIZE_TOKENS=$MAX_RANK_MICROBATCH_SIZE_TOKENS"
echo "LEARNING_RATE=$LEARNING_RATE"

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
    --max_rank_microbatch_size_tokens="$MAX_RANK_MICROBATCH_SIZE_TOKENS" \
    --model_name="olmo3-7b" \
    --dataset_path="$DATASET_PATH" \
    --train_module.optim.lr="$LEARNING_RATE" \
    --trainer.max_duration.value=2 \
    --trainer.max_duration.unit=epochs \
    --trainer.callbacks.wandb.enabled="$WANDB_ENABLED" \
    --trainer.callbacks.wandb.project="$WANDB_PROJECT" \
    --trainer.callbacks.wandb.entity="$WANDB_ENTITY" \
    --trainer.callbacks.wandb.name="$RUN_NAME" \
    --trainer.callbacks.wandb.tags="$WANDB_TAGS_JSON" \
    --trainer.callbacks.wandb.cancel_check_interval="$WANDB_CANCEL_CHECK_INTERVAL" \
    --trainer.callbacks.wandb.cancel_tags="$WANDB_CANCEL_TAGS_JSON" \
    --trainer.callbacks.checkpointer.ephemeral_save_interval=150 \
    --save_tokenizer=True \
    --budget=unused \
    --workspace=unused \
    "${EXTRA_ACCELERATE_ARGS[@]}"
    # seed is controlled by `init_seed` inside the SFT script config; avoid passing unsupported overrides here
