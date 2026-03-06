#!/bin/bash
#SBATCH --job-name=eval-test
#SBATCH --partition=testdlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/multilingual_eu/logs/eval_test_%j.log

# Quick smoke test: 5 prompts to verify judge model works
set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
source "$PROJECT_ROOT/.venv/bin/activate"

export OPENJURY_DATA="$PROJECT_ROOT/data/openjury-eval-data"
export PYTHONPATH="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"

# Symlinks for short model names (avoids path-too-long errors in OpenJury)
SYMLINK_DIR="$PROJECT_ROOT/models/eval"
mkdir -p "$SYMLINK_DIR"
TEMP_BASELINE="$SYMLINK_DIR/instruct-baseline"
TEMP_OURS="$SYMLINK_DIR/eu-test"
rm -f "$TEMP_BASELINE" "$TEMP_OURS"
ln -sf "$PROJECT_ROOT/models/baselines/Olmo-3-7B-Instruct-SFT" "$TEMP_BASELINE"
ln -sf "$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-eu-A1-90en-hf/step238" "$TEMP_OURS"

cd "$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"

python openjury/generate_and_evaluate.py \
    --dataset m-arena-hard-EU \
    --model_A "VLLM/$TEMP_BASELINE" \
    --model_B "VLLM/$TEMP_OURS" \
    --judge_model "VLLM/Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --n_instructions 5 \
    --result_folder "$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury/results/multilingual_eu/eval-test-delete-me" \
    --max_out_tokens_models 8192 \
    --max_out_tokens_judge 8192 \
    --truncate_all_input_chars 32768 \
    --swap_mode fixed \
    --eval_mode winrate \
    --provide_explanation \
    --ignore_cache

echo "TEST PASSED"
