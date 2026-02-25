#!/usr/bin/env bash
#SBATCH --job-name=olmo3-7b-instruct-sft
#SBATCH --partition=accelerated-h200
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/fr_ff1042-oellm/open-instruct/oellm/horeka/logs/%A_%a.%x.%N.out
#SBATCH --error=/hkfs/work/workspace/scratch/fr_ff1042-oellm/open-instruct/oellm/horeka/logs/%A_%a.%x.%N.err
#SBATCH --account=hk-project-p0024002
#SBATCH --array=0-9%1

# HoreKa Instruct SFT training script for OLMo-3-7B
# Continues from Think SFT checkpoint (think → instruct pipeline, per paper Section 5.2.2)
# 2 nodes × 4 H200 GPUs = 8 GPUs total
# LR=8e-5, 1M token batch, 32K seq len, 2 epochs (paper Table 47)
#
# Usage:
#   sbatch oellm/horeka/train_instruct_sft_horeka.sh

set -euo pipefail

# === HoreKa workspace paths ===
WS="/hkfs/work/workspace/scratch/fr_ff1042-oellm"
PROJECT_ROOT="${WS}/open-instruct"
OLMOCORE_PATH="${WS}/OLMo-core"

if [ -f "$PROJECT_ROOT/.env" ]; then
  export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

module purge
module use /software/easybuild/modules/all/
module load Python/3.12.3-GCCcore-13.3.0
module load devel/cuda/12.4

source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

# === Training configuration ===
# Instruct SFT starts from Think SFT checkpoint (warm-start, per paper Section 5.2.2)
RUN_NAME="${RUN_NAME:-dolci-instruct-sft-v2-horeka}"
CLUSTER_NAME="slurm"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/baseline_reproduction/dolci_instruct_sft_tokenized_v2}"
BASE_CKPT="${BASE_CKPT:-${PROJECT_ROOT}/checkpoints/fr_ff1042/olmo3-7b-sft/dolci-think-sft-v2-horeka/step42856}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/.cache}"
LEARNING_RATE="${LEARNING_RATE:-8e-5}"  # 8e-5 for Instruct (from paper Table 47)
SEQ_LEN="${SEQ_LEN:-32768}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((SEQ_LEN * 32))}"
MAX_RANK_MICROBATCH_SIZE_TOKENS="${MAX_RANK_MICROBATCH_SIZE_TOKENS:-24576}"

# === Multi-node ===
NUM_MACHINES=${SLURM_NNODES:-2}
GPUS_PER_NODE=4
TOTAL_GPUS=$((GPUS_PER_NODE * NUM_MACHINES))

# === Environment ===
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OLMO_SHARED_FS=1
export NCCL_IB_DISABLE=0             # Enable InfiniBand for inter-node
export NCCL_IB_HCA=mlx5              # Use Mellanox ConnectX HCA
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"  # INFO for first run to diagnose issues

# === Smoke test ===
TEST_RUN="${TEST_RUN:-false}"
TEST_STEPS="${TEST_STEPS:-20}"
EXTRA_ARGS=()
if [[ "${TEST_RUN}" == "true" ]]; then
  echo "TEST_RUN=true: limiting to ${TEST_STEPS} steps."
  EPHEMERAL_INTERVAL=$((TEST_STEPS / 2))
  EXTRA_ARGS+=(
    "--trainer.max_duration.value=${TEST_STEPS}"
    "--trainer.max_duration.unit=steps"
    "--trainer.callbacks.checkpointer.save_interval=${TEST_STEPS}"
    "--trainer.callbacks.checkpointer.ephemeral_save_interval=${EPHEMERAL_INTERVAL}"
  )
fi

# === W&B ===
WANDB_ENABLED="${WANDB_ENABLED:-auto}"
if [[ "${WANDB_ENABLED}" == "auto" ]]; then
  [[ -n "${WANDB_API_KEY:-}" ]] && WANDB_ENABLED=true || WANDB_ENABLED=false
fi
WANDB_PROJECT="${WANDB_PROJECT:-olmo-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}"
WANDB_TAGS_JSON="${WANDB_TAGS_JSON:-[\"dolci\",\"instruct\",\"sft\",\"7b\",\"horeka\"]}"
WANDB_CANCEL_CHECK_INTERVAL="${WANDB_CANCEL_CHECK_INTERVAL:-1000000000}"
WANDB_CANCEL_TAGS_JSON="${WANDB_CANCEL_TAGS_JSON:-[]}"

mkdir -p "$PROJECT_ROOT/oellm/horeka/logs" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_HUB_CACHE"

# === Master node ===
MAIN_PROCESS_IP=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MAIN_PROCESS_PORT=$((29500 + (${SLURM_JOB_ID} % 10000)))
export MASTER_ADDR=$MAIN_PROCESS_IP
export MASTER_PORT=$MAIN_PROCESS_PORT

echo "=========================================="
echo "HoreKa Instruct SFT Configuration"
echo "=========================================="
echo "RUN_NAME=$RUN_NAME"
echo "DATASET=$DATASET_PATH"
echo "BASE_CKPT=$BASE_CKPT"
echo "NUM_MACHINES=$NUM_MACHINES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "TOTAL_GPUS=$TOTAL_GPUS"
echo "SEQ_LEN=$SEQ_LEN"
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "MAIN_PROCESS_IP=$MAIN_PROCESS_IP"
echo "MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
echo "NODELIST=$SLURM_NODELIST"
echo "=========================================="

srun bash -c "accelerate launch \
  --mixed_precision bf16 \
  --num_machines $NUM_MACHINES \
  --num_processes $TOTAL_GPUS \
  --machine_rank \$SLURM_NODEID \
  --main_process_ip $MAIN_PROCESS_IP \
  --main_process_port $MAIN_PROCESS_PORT \
  --use_deepspeed \
  --deepspeed_config_file configs/ds_configs/stage2_accelerate.conf \
  --deepspeed_multinode_launcher standard \
  ${OLMOCORE_PATH}/src/scripts/train/sft/OLMo-sft.py train \
    $RUN_NAME \
    $BASE_CKPT \
    $CLUSTER_NAME \
    --seq_len=$SEQ_LEN \
    --num_nodes=$NUM_MACHINES \
    --global_batch_size=$GLOBAL_BATCH_SIZE \
    --max_rank_microbatch_size_tokens=$MAX_RANK_MICROBATCH_SIZE_TOKENS \
    --model_name=olmo3-7b \
    --dataset_path=$DATASET_PATH \
    --train_module.optim.lr=$LEARNING_RATE \
    --trainer.max_duration.value=2 \
    --trainer.max_duration.unit=epochs \
    --trainer.callbacks.wandb.enabled=$WANDB_ENABLED \
    --trainer.callbacks.wandb.project=$WANDB_PROJECT \
    --trainer.callbacks.wandb.entity=$WANDB_ENTITY \
    --trainer.callbacks.wandb.name=$RUN_NAME \
    --trainer.callbacks.wandb.tags='$WANDB_TAGS_JSON' \
    --trainer.callbacks.wandb.cancel_check_interval=$WANDB_CANCEL_CHECK_INTERVAL \
    --trainer.callbacks.wandb.cancel_tags='$WANDB_CANCEL_TAGS_JSON' \
    --trainer.callbacks.checkpointer.ephemeral_save_interval=150 \
    --save_tokenizer=True \
    --budget=unused \
    --workspace=unused \
    ${EXTRA_ARGS[@]:-}"
