#!/usr/bin/env bash
set -euo pipefail

# Verify checksums between local and remote
REMOTE_HOST="ferreira@aadlogin.informatik.uni-freiburg.de"
DATASET_ID="allenai/Dolci-Instruct-SFT"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_DATA_DIR="${REPO_ROOT}/.cache/huggingface/datasets/$DATASET_ID/data"
REMOTE_DATA_DIR="~/data/downloads/$DATASET_ID/data"

echo "Calculating local checksums..."
cd "$LOCAL_DATA_DIR"
find . -type f -name "*.parquet" -print0 | xargs -0 sha256sum | sort > /tmp/local_checksums.txt

echo "Calculating remote checksums..."
ssh "$REMOTE_HOST" "cd $REMOTE_DATA_DIR && find . -type f -name '*.parquet' -print0 | xargs -0 sha256sum | sort" > /tmp/remote_checksums.txt

echo "Comparing..."
if diff /tmp/local_checksums.txt /tmp/remote_checksums.txt; then
    echo "SUCCESS: All checksums match between local and remote."
else
    echo "FAILURE: Checksums do not match!"
    exit 1
fi

