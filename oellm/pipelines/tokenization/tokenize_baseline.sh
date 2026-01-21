#!/usr/bin/env bash
#SBATCH --job-name=tokenize-sft-mixture
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/pipelines/tokenization/logs/tokenize_%j.log

# Tokenize SFT datasets for OLMo-core training
#
# H200 nodes: 8 GPUs, 1500GB RAM, 384 CPUs per node
# Requesting 1 GPU gives us ~187GB RAM and ~48 CPUs proportionally
#
# Reads a mixture config file to determine which datasets to include and their weights.
# Output directory is automatically derived from config file name.
#
# Usage:
#   # Tokenize for baseline reproduction (100% English)
#   sbatch tokenize_baseline.sh
#
#   # Use a specific config file
#   CONFIG_FILE=/path/to/config.yaml sbatch tokenize_baseline.sh
#
#   # Request more resources (2 GPUs = ~375GB RAM, ~96 CPUs)
#   sbatch --gpus=2 tokenize_baseline.sh

set -euo pipefail

# Load env vars from .env if present (for PROJECT_ROOT, proxies, etc.)
if [ -f /work/dlclarge2/ferreira-oellm/open-instruct/.env ]; then
    export $(grep -v '^#' /work/dlclarge2/ferreira-oellm/open-instruct/.env | xargs)
fi

# Auto-detect available CPUs from SLURM, fallback to 48 (1 GPU worth)
# Use 75% of allocated CPUs to leave headroom for system processes
ALLOCATED_CPUS="${SLURM_CPUS_ON_NODE:-48}"
WORKER_COUNT=$((ALLOCATED_CPUS * 3 / 4))
export BEAKER_ASSIGNED_CPU_COUNT="${BEAKER_ASSIGNED_CPU_COUNT:-$WORKER_COUNT}"

# Configuration (PROJECT_ROOT comes from .env)
PROJECT_ROOT="${PROJECT_ROOT:-/work/dlclarge2/ferreira-oellm/open-instruct}"
SCRIPT_DIR="$PROJECT_ROOT/oellm/pipelines/tokenization"

# Paths
PREPROCESSED_DIR="${PREPROCESSED_DIR:-$PROJECT_ROOT/data/datasets_mixture_sft_preprocessed}"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/mixture_baseline.yaml}"

# Derive output directory from config file name
# mixture_all.yaml -> datasets_mixture_sft_tokenized
# mixture_safety.yaml -> datasets_mixture_sft_tokenized_safety
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
if [[ "$CONFIG_BASENAME" == "mixture_all" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/datasets_mixture_sft_tokenized}"
else
    # Extract suffix after "mixture_"
    CONFIG_SUFFIX="${CONFIG_BASENAME#mixture_}"
    OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/datasets_mixture_sft_tokenized_${CONFIG_SUFFIX}}"
fi

# Tokenizer - using OLMo-3-7B-Instruct-SFT
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Instruct-SFT}"

# Processing options
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32768}"

# HuggingFace cache
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# Create directories
mkdir -p "$OUTPUT_DIR" "$SCRIPT_DIR/logs"

# Activate environment
source "$PROJECT_ROOT/.venv/bin/activate"

echo "=============================================="
echo "SFT Dataset Tokenization"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Preprocessed dir: $PREPROCESSED_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Config file: $CONFIG_FILE"
echo "Tokenizer: $TOKENIZER"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo ""
echo "Resources allocated:"
echo "  CPUs: ${SLURM_CPUS_ON_NODE:-unknown}"
echo "  Memory: $(free -h | awk '/^Mem:/{print $2}') available on node"
echo "  Workers: $BEAKER_ASSIGNED_CPU_COUNT"
echo "=============================================="
echo ""

# Check config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Create a config file with your dataset selection:"
    echo "  cp $SCRIPT_DIR/mixture_config.yaml $CONFIG_FILE"
    exit 1
fi

# Parse config file and build dataset_mixer_list arguments
echo "Reading config: $CONFIG_FILE"
echo ""

# Verify datasets exist and build args
DATASET_ARGS=$(python3 << EOF
import yaml
import sys
import os

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

datasets = config.get('datasets', [])
if not datasets:
    print('ERROR: No datasets found in config file', file=sys.stderr)
    sys.exit(1)

args = []
missing = []
print("Selected datasets:", file=sys.stderr)
for ds in datasets:
    name = ds['name']
    weight = ds.get('weight', 1.0)
    path = os.path.join('$PREPROCESSED_DIR', name)

    if not os.path.exists(path):
        missing.append(name)
        print(f"  [MISSING] {name} (weight: {weight})", file=sys.stderr)
    else:
        print(f"  [OK] {name} (weight: {weight})", file=sys.stderr)
        args.append(path)
        args.append(str(weight))

if missing:
    print(f"\nERROR: {len(missing)} dataset(s) not found. Run download_datasets.sh first.", file=sys.stderr)
    sys.exit(1)

print(f"\nTotal: {len(datasets)} dataset(s)", file=sys.stderr)
print(' '.join(args))
EOF
)

if [ $? -ne 0 ]; then
    echo ""
    echo "Failed to parse config or missing datasets."
    exit 1
fi

echo ""

# Run tokenization
echo "=============================================="
echo "Running tokenization..."
echo "=============================================="
echo ""

python "$PROJECT_ROOT/scripts/data/convert_sft_data_for_olmocore.py" \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list $DATASET_ARGS \
    --output_dir "$OUTPUT_DIR" \
    --chat_template_name olmo \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --visualize

echo ""
echo "=============================================="
echo "TOKENIZATION COMPLETE!"
echo "=============================================="
echo ""
echo "Tokenized data saved to: $OUTPUT_DIR"
echo ""
echo "To train with OLMo-core:"
echo "  DATASET_PATH=$OUTPUT_DIR sbatch oellm/train/train_think_sft_dolci_7b_slurm.sh"
