#!/bin/bash
# Setup script for benchmark evaluation
# Run this on the cluster to install OpenJury and dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"

echo "=== Setting up Benchmark Evaluation ==="
echo ""

# 1. Clone OpenJury if not present
if [ ! -d "$SCRIPT_DIR/OpenJury" ]; then
    echo "1. Cloning OpenJury..."
    cd "$SCRIPT_DIR"
    git clone https://github.com/OpenEuroLLM/OpenJury.git
    echo "   Done!"
else
    echo "1. OpenJury already cloned"
fi

# 2. Install dependencies
echo ""
echo "2. Installing dependencies into project venv..."
cd "$PROJECT_ROOT"
source .venv/bin/activate

# OpenJury dependencies (langchain for LLM providers)
# Note: OpenJury requires langchain 0.3.x, not 1.x
pip install 'langchain>=0.3.27,<1.0.0' 'langchain-openai>=0.3.32,<1.0.0' 'langchain-community>=0.3.29,<1.0.0' 2>/dev/null || {
    echo "   Note: Some langchain packages may already be installed"
}

# FlashInfer for faster vLLM sampling (optional but recommended)
pip install flashinfer-python 2>/dev/null || {
    echo "   Note: FlashInfer installation skipped (optional)"
}

echo "   Done!"

# 3. Pre-download datasets
echo ""
echo "3. Pre-downloading evaluation datasets..."
export OPENJURY_EVAL_DATA="$PROJECT_ROOT/data/openjury-eval-data"
python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/OpenJury')
try:
    from openjury.utils import download_all
    download_all()
    print('   Datasets downloaded to: $OPENJURY_EVAL_DATA')
except Exception as e:
    print(f'   Warning: Could not download datasets: {e}')
    print('   Datasets will be downloaded on first use.')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Install slurmpilot on your LOCAL machine: pip install slurmpilot"
echo "  2. Configure slurmpilot: sp --init"
echo "  3. Run evaluation: python launch_evaluation.py"

