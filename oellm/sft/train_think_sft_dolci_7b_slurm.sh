#!/usr/bin/env bash
#SBATCH --job-name=dolci-think-sft
#SBATCH --partition=alldlc2_gpu-h200 #testdlc2_gpu-h200#alldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/olmo3-7b-think-sft/%j.%x.%N.out
#SBATCH --error=slurm_logs/olmo3-7b-think-sft/%j.%x.%N.err

set -euo pipefail

# load env vars from .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

source /work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/activate

# Ensure we use the OLMo-core source checkout (with local patches) instead of any pip-installed version.
OLMOCORE_PATH="/work/dlclarge2/ferreira-oellm/OLMo-core"
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-dolci-think-sft}"
CLUSTER_NAME="slurm"
GPUS="${GPUS:-8}"
DATASET_PATH="${DATASET_PATH:-/work/dlclarge2/ferreira-oellm/open-instruct/data/dolci_think_sft_tokenized}"
BASE_CKPT="${BASE_CKPT:-/work/dlclarge2/ferreira-oellm/open-instruct/models/Olmo-3-1025-7B-olmocore}"
CACHE_DIR="${CACHE_DIR:-/work/dlclarge2/ferreira-oellm/open-instruct/.cache}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
SEQ_LEN="${SEQ_LEN:-32768}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((SEQ_LEN * 32))}"
SEED="${SEED:-42}"

export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

mkdir -p slurm_logs/dolci-think-sft "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_HUB_CACHE"

echo "RUN_NAME=$RUN_NAME"
echo "DATASET=$DATASET_PATH"
echo "BASE_CKPT=$BASE_CKPT"
echo "GPUS=$GPUS"
echo "SEQ_LEN=$SEQ_LEN"
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "LEARNING_RATE=$LEARNING_RATE"

NUM_GPUS=$GPUS
NUM_MACHINES=1
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
MAIN_PROCESS_PORT=29500

srun accelerate launch \
  --mixed_precision bf16 \
  --num_machines $NUM_MACHINES \
  --num_processes $NUM_GPUS \
  --machine_rank $MACHINE_RANK \
  --main_process_ip $MAIN_PROCESS_IP \
  --main_process_port $MAIN_PROCESS_PORT \
  --use_deepspeed \
  --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
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
    --trainer.callbacks.wandb.enabled=False \
    --save_tokenizer=True \
    --budget=unused \
    --workspace=unused \
    # seed is controlled by `init_seed` inside the SFT script config; avoid passing unsupported overrides here
