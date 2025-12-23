#!/usr/bin/env bash
#SBATCH --job-name=olmo3-7b-dolci-think-sft-multinode
#SBATCH --partition=alldlc2_gpu-h200 #testdlc2_gpu-h200 #mldlc2_gpu-h200
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/olmo3-7b-dolci-think-sft-multinode/%A_%a.%x.%N.out
#SBATCH --error=slurm_logs/olmo3-7b-dolci-think-sft-multinode/%A_%a.%x.%N.err
#SBATCH --array=0-9%1

set -euo pipefail

# load env vars from .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

source /work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/activate

# Ensure we use the OLMo-core source checkout (with local patches) instead of any pip-installed version.
OLMOCORE_PATH="/work/dlclarge2/ferreira-oellm/OLMo-core"
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-dolci-think-sft-multinode}"
CLUSTER_NAME="slurm"
DATASET_PATH="${DATASET_PATH:-/work/dlclarge2/ferreira-oellm/open-instruct/data/dolci_think_sft_tokenized}"
BASE_CKPT="${BASE_CKPT:-/work/dlclarge2/ferreira-oellm/open-instruct/models/Olmo-3-1025-7B-olmocore}"
CACHE_DIR="${CACHE_DIR:-/work/dlclarge2/ferreira-oellm/open-instruct/.cache}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
SEQ_LEN="${SEQ_LEN:-32768}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((SEQ_LEN * 32))}" # 1M tokens (per baseline paper)
SEED="${SEED:-42}"

# Simple explicit setup - read directly from Slurm allocation
NUM_MACHINES=${SLURM_NNODES:-1}
GPUS_PER_NODE=4  # Match --gres=gpu:4 in SBATCH header
TOTAL_GPUS=$((GPUS_PER_NODE * NUM_MACHINES))

export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OLMO_SHARED_FS=1  # Tell OLMo-core that we have a shared filesystem for multi-node checkpointing

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
# Tags must be a *list* for the OLMo-core config parser. Use a JSON-ish list string.
WANDB_TAGS_JSON="${WANDB_TAGS_JSON:-[\"dolci\",\"think\",\"sft\",\"7b\"]}"
# W&B "cancel by tag" checks require W&B API connectivity; disable by default to avoid 30s stalls on clusters w/o egress.
WANDB_CANCEL_CHECK_INTERVAL="${WANDB_CANCEL_CHECK_INTERVAL:-1000000000}"
WANDB_CANCEL_TAGS_JSON="${WANDB_CANCEL_TAGS_JSON:-[]}"

mkdir -p slurm_logs/olmo3-7b-dolci-think-sft-multinode "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_HUB_CACHE"

echo "=========================================="
echo "Multi-Node Training Configuration"
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

# Get the main process IP from the first node in the allocation
if [ $NUM_MACHINES -gt 1 ]; then
    MAIN_PROCESS_IP=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
    echo "MAIN_PROCESS_IP=$MAIN_PROCESS_IP"
    echo "NODELIST=$SLURM_NODELIST"
else
    MAIN_PROCESS_IP=localhost
    echo "Single-node training on localhost"
fi

# Use job ID to generate unique port (29500 + last 4 digits of job ID)
# This avoids port conflicts between concurrent jobs
MAIN_PROCESS_PORT=$((29500 + (${SLURM_JOB_ID} % 10000)))
echo "MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT (derived from job ID ${SLURM_JOB_ID})"
echo "=========================================="

# Set master address and port for PyTorch distributed
export MASTER_ADDR=$MAIN_PROCESS_IP
export MASTER_PORT=$MAIN_PROCESS_PORT

# Launch configuration
if [ $NUM_MACHINES -eq 1 ]; then
    # Single-node: simpler launch
    srun accelerate launch \
      --mixed_precision bf16 \
      --num_machines 1 \
      --num_processes $GPUS_PER_NODE \
      --machine_rank 0 \
      --main_process_ip localhost \
      --main_process_port $MAIN_PROCESS_PORT \
      --use_deepspeed \
      --deepspeed_config_file configs/ds_configs/stage2_accelerate.conf \
      --deepspeed_multinode_launcher standard \
      "${OLMOCORE_PATH}/src/scripts/train/sft/OLMo-sft.py" train \
        "$RUN_NAME" \
        "$BASE_CKPT" \
        "$CLUSTER_NAME" \
        --seq_len="$SEQ_LEN" \
        --num_nodes=1 \
        --global_batch_size="$GLOBAL_BATCH_SIZE" \
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
        --save_tokenizer=True \
        --budget=unused \
        --workspace=unused
else
    # Multi-node: use detected configuration
    # SLURM_NODEID provides the rank of current node (0, 1, 2, ...)
    # Each node launches only GPUS_PER_NODE processes (not TOTAL_GPUS)
    # Use bash -c to evaluate SLURM_NODEID on each node
    srun bash -c "accelerate launch \
      --mixed_precision bf16 \
      --num_machines $NUM_MACHINES \
      --num_processes $GPUS_PER_NODE \
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
        --save_tokenizer=True \
        --budget=unused \
        --workspace=unused"
fi
# seed is controlled by `init_seed` inside the SFT script config; avoid passing unsupported overrides here

