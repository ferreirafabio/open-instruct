#!/bin/bash
# Rsync data to horeka workspace. Run from kislurm.
# Uses SSH proxy via aadlogin. Requires OTP+password handled by expect_rsync.py wrapper.

set -e

WS_REMOTE="/hkfs/work/workspace/scratch/fr_ff1042-oellm/open-instruct"
LOCAL="/work/dlclarge2/ferreira-oellm/open-instruct"
SSH_CMD='ssh -o ProxyCommand="ssh -o ControlPath=/tmp/ssh-mux/aadlogin -W %h:%p ferreira@aadlogin.informatik.uni-freiburg.de"'

echo "=== Syncing v2 tokenized datasets ==="

echo "[1/4] dolci_think_sft_tokenized_v2 (105GB)..."
rsync -avz --progress \
    -e "$SSH_CMD" \
    "$LOCAL/data/baseline_reproduction/dolci_think_sft_tokenized_v2/" \
    "fr_ff1042@horeka.scc.kit.edu:$WS_REMOTE/data/baseline_reproduction/dolci_think_sft_tokenized_v2/"

echo "[2/4] dolci_instruct_sft_tokenized_v2 (8GB)..."
rsync -avz --progress \
    -e "$SSH_CMD" \
    "$LOCAL/data/baseline_reproduction/dolci_instruct_sft_tokenized_v2/" \
    "fr_ff1042@horeka.scc.kit.edu:$WS_REMOTE/data/baseline_reproduction/dolci_instruct_sft_tokenized_v2/"

echo "[3/4] Base model checkpoint (28GB)..."
rsync -avz --progress \
    -e "$SSH_CMD" \
    "$LOCAL/models/baselines/Olmo-3-1025-7B-olmocore/" \
    "fr_ff1042@horeka.scc.kit.edu:$WS_REMOTE/models/baselines/Olmo-3-1025-7B-olmocore/"

echo "[4/4] OLMo-core code (47MB)..."
rsync -avz --progress \
    -e "$SSH_CMD" \
    "/work/dlclarge2/ferreira-oellm/OLMo-core/" \
    "fr_ff1042@horeka.scc.kit.edu:$WS_REMOTE/../OLMo-core/"

echo "=== ALL TRANSFERS COMPLETE ==="
