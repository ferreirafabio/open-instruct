#!/usr/bin/env bash
#SBATCH --job-name=olmo3-7b-think-sft
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=460G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/olmo3-7b-think-sft/%j.%x.%N.out
#SBATCH --error=slurm_logs/olmo3-7b-think-sft/%j.%x.%N.err

set -euo pipefail

# optionally load env vars from .env similar to user template
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

source /work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/activate

RUN_NAME="${RUN_NAME:-olmo3-7b-sft}"
DATASET_PATH="${DATASET_PATH:-/work/dlclarge2/ferreira-oellm/open-instruct/data/dolci_think_sft_tokenized}"
BASE_CKPT="${BASE_CKPT:-/work/dlclarge2/ferreira-oellm/open-instruct/models/Olmo-3-1025-7B}"
CACHE_DIR="${CACHE_DIR:-/work/dlclarge2/ferreira-oellm/open-instruct/.cache}"
SEED="${SEED:-42}"

export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

mkdir -p slurm_logs/olmo3-7b-think-sft "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_HUB_CACHE"

echo "Training run: $RUN_NAME"
echo "Dataset path: $DATASET_PATH"
echo "Base checkpoint: $BASE_CKPT"
echo "HF_CACHE: $HF_HOME"

NUM_GPUS=8
NUM_MACHINES=1
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
MAIN_PROCESS_PORT=29500
TOKEN_BATCH_SIZE=32768
TOKENS_PER_BATCH=1048576 # 1M tokens

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
  src/scripts/train/sft/OLMo-sft.py train \
    "$RUN_NAME" \
    "$BASE_CKPT" \
    alldlc2_gpu-h200 \
    --seq_len=$TOKEN_BATCH_SIZE \
    --num_nodes=$NUM_MACHINES \
    --global_batch_size=$TOKENS_PER_BATCH \
    --model_name="olmo3-7b" \
    --dataset_path="$DATASET_PATH" \
    --train_module.optim.lr=1e-4 \
    --trainer.max_duration.value=2 \
    --trainer.max_duration.unit=epoch \
    --trainer.callbacks.wandb.enabled=False \
    --save_tokenizer=True \
    --budget=unused \
    --workspace=unused \
    --trainer.seed="$SEED" \
    --oe_eval_max_length=32768

