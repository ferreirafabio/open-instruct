#!/bin/bash
#SBATCH --job-name=llm-judge-eval
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/logs/eval_%j.log

# LLM-Judge evaluation using OpenJury
# Compares trained model against baseline using automated judges
#
# Usage:
#   sbatch run_evaluation.sh                    # Run with defaults
#   sbatch run_evaluation.sh alpaca-eval        # Specific dataset
#   sbatch run_evaluation.sh arena-hard 100     # Dataset + num instructions

set -euo pipefail

# Configuration
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
OPENJURY_DIR="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Models to compare
BASELINE="$PROJECT_ROOT/models/baselines/Olmo-3-7B-Instruct-SFT"
TRAINED="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf"

# Judge model (runs on same GPU)
JUDGE_MODEL="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"

# Arguments
DATASET="${1:-alpaca-eval}"
N_INSTRUCTIONS="${2:-100}"  # Start small for testing

# Environment
export OPENJURY_EVAL_DATA="$PROJECT_ROOT/data/openjury-eval-data"
export PYTHONPATH="$OPENJURY_DIR:${PYTHONPATH:-}"

echo "=============================================="
echo "LLM-Judge Evaluation"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Instructions: $N_INSTRUCTIONS"
echo "Baseline: $BASELINE"
echo "Trained: $TRAINED"
echo "Judge: $JUDGE_MODEL"
echo "=============================================="
echo ""

cd "$OPENJURY_DIR"

# Run evaluation
$VENV_PYTHON openjury/generate_and_evaluate.py \
    --dataset "$DATASET" \
    --model_A "VLLM/$BASELINE" \
    --model_B "VLLM/$TRAINED" \
    --judge_model "VLLM/$JUDGE_MODEL" \
    --n_instructions "$N_INSTRUCTIONS"

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "=============================================="

