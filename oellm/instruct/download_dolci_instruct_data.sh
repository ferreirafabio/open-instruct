#!/usr/bin/env bash
set -euo pipefail

# Usage: ./download_dolci_instruct_data.sh [--remote]
# If --remote is passed, it uses the jump host to download and then rsyncs back.

REMOTE_HOST="ferreira@aadlogin.informatik.uni-freiburg.de"
DATASET_ID="allenai/Dolci-Instruct-SFT"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_CACHE_DIR="${REPO_ROOT}/.cache/huggingface/datasets"

if [[ "${1:-}" == "--remote" ]]; then
    echo "Using remote host $REMOTE_HOST to download data..."
    
    REMOTE_DIR="~/data/downloads/$DATASET_ID"
    
    # Combined SSH call to detect environment and download
    echo "Running download on remote machine (detecting environment)..."
    ssh "$REMOTE_HOST" "
        mkdir -p $REMOTE_DIR && \
        # Try to find a newer python if possible
        PYTHON_CMD='python3'
        if command -v python3.10 >/dev/null 2>&1; then PYTHON_CMD='python3.10';
        elif command -v python3.9 >/dev/null 2>&1; then PYTHON_CMD='python3.9';
        elif command -v python3.8 >/dev/null 2>&1; then PYTHON_CMD='python3.8';
        fi
        
        echo \"Using \$PYTHON_CMD\"
        
        # Try to install huggingface_hub without hf_transfer (simpler)
        \$PYTHON_CMD -m pip install --user huggingface_hub >/dev/null 2>&1 || echo 'pip install failed, trying with existing environment'
        
        \$PYTHON_CMD -c \"
import os, sys
try:
    from huggingface_hub import snapshot_download
    print('Starting download of $DATASET_ID...')
    snapshot_download(
        repo_id='$DATASET_ID',
        repo_type='dataset',
        local_dir=os.path.expanduser('$REMOTE_DIR'),
        local_dir_use_symlinks=False
    )
    print('Download complete.')
except ImportError:
    # Fallback to git lfs if python fails
    print('huggingface_hub not found, checking for git-lfs...')
    exit(1)
\" || (
    if command -v git-lfs >/dev/null 2>&1; then
        echo 'Using git-lfs to download...'
        git clone https://huggingface.co/datasets/$DATASET_ID $REMOTE_DIR
    else
        echo 'Error: Neither huggingface_hub nor git-lfs found on remote.'
        exit 1
    fi
) "
    
    # 3. Rsync the data back
    echo "Syncing data back to $LOCAL_CACHE_DIR..."
    mkdir -p "$LOCAL_CACHE_DIR/$DATASET_ID"
    rsync -avzP "$REMOTE_HOST:${REMOTE_DIR}/" "$LOCAL_CACHE_DIR/$DATASET_ID/"
    
    echo "Remote download and sync complete."
else
    echo "Downloading $DATASET_ID locally to $LOCAL_CACHE_DIR..."
    mkdir -p "$LOCAL_CACHE_DIR"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    huggingface-cli download --repo-type dataset "$DATASET_ID" --local-dir "$LOCAL_CACHE_DIR/$DATASET_ID" --local-dir-use-symlinks False
fi


