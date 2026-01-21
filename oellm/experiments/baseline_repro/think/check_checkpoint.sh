#!/usr/bin/env bash
#SBATCH --job-name=check-ckpt
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --gpus=6
#SBATCH --time=00:30:00
#SBATCH --output=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/baseline_repro/think/logs/%j.%x.out
#SBATCH --error=/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/baseline_repro/think/logs/%j.%x.err

# Check checkpoint integrity
CHECKPOINT="/work/dlclarge2/ferreira-oellm/open-instruct/data/baseline_reproduction/dolci_think_sft_tokenized_v2/_checkpoint.json"

echo "=== Checkpoint Analysis ==="
echo "File: $CHECKPOINT"
echo "Size: $(du -h $CHECKPOINT | cut -f1)"
echo ""

python3 << 'EOF'
import os
import time
from tqdm import tqdm

path = "/work/dlclarge2/ferreira-oellm/open-instruct/data/baseline_reproduction/dolci_think_sft_tokenized_v2/_checkpoint.json"
file_size = os.path.getsize(path)

# Try orjson (10x faster) or fall back to json
try:
    import orjson
    use_orjson = True
    print("Using orjson (fast)")
except ImportError:
    import json
    use_orjson = False
    print("Using json (slow)")

print(f"\nLoading {file_size / (1024**3):.1f} GB checkpoint...")
start = time.time()

# Read file with progress bar
chunk_size = 100 * 1024 * 1024  # 100MB chunks
data = bytearray()
with open(path, 'rb') as f:
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading") as pbar:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            data.extend(chunk)
            pbar.update(len(chunk))

print("Parsing JSON...")
parse_start = time.time()
if use_orjson:
    ckpt = orjson.loads(bytes(data))
else:
    ckpt = json.loads(data.decode('utf-8'))
parse_time = time.time() - parse_start
print(f"Parsed in {parse_time:.1f} seconds")

elapsed = time.time() - start
print(f"Total load time: {elapsed:.1f} seconds")

print(f"\n=== Checkpoint Contents ===")
print(f"samples_processed: {ckpt['samples_processed']:,}")
print(f"token_ids length: {len(ckpt['token_ids']):,}")
print(f"labels_mask length: {len(ckpt['labels_mask']):,}")
print(f"document_boundaries length: {len(ckpt['document_boundaries']):,}")
print(f"current_position: {ckpt['current_position']:,}")
print(f"num_samples_skipped: {ckpt['num_samples_skipped']:,}")

print(f"\n=== Derived Stats ===")
print(f"Avg tokens per sample: {len(ckpt['token_ids']) / ckpt['samples_processed']:.1f}")
print(f"Trainable tokens: {sum(ckpt['labels_mask']):,}")
print(f"Trainable ratio: {sum(ckpt['labels_mask']) / len(ckpt['labels_mask']) * 100:.1f}%")

print(f"\n=== Per-Dataset Stats ===")
for ds, count in ckpt.get('per_dataset_counts', {}).items():
    tokens = ckpt.get('per_dataset_tokens', {}).get(ds, 0)
    trainable = ckpt.get('per_dataset_trainable_tokens', {}).get(ds, 0)
    filtered = ckpt.get('per_dataset_filtered', {}).get(ds, 0)
    print(f"  {ds}: {count:,} samples, {tokens:,} tokens, {trainable:,} trainable, {filtered:,} filtered")

print(f"\n=== Validation ===")
# Check consistency
errors = []
if len(ckpt['token_ids']) != len(ckpt['labels_mask']):
    errors.append(f"token_ids ({len(ckpt['token_ids'])}) != labels_mask ({len(ckpt['labels_mask'])})")
if len(ckpt['document_boundaries']) != ckpt['samples_processed']:
    errors.append(f"document_boundaries ({len(ckpt['document_boundaries'])}) != samples_processed ({ckpt['samples_processed']})")
if ckpt['current_position'] != len(ckpt['token_ids']):
    errors.append(f"current_position ({ckpt['current_position']}) != token_ids length ({len(ckpt['token_ids'])})")

if errors:
    print("ERRORS FOUND:")
    for e in errors:
        print(f"  - {e}")
else:
    print("✅ Checkpoint appears valid!")

# Check last few boundaries
print(f"\n=== Last 5 Document Boundaries ===")
for i, (start, end) in enumerate(ckpt['document_boundaries'][-5:]):
    print(f"  Doc {ckpt['samples_processed'] - 5 + i}: [{start:,}, {end:,}] (len={end-start})")
EOF

echo ""
echo "=== Done ==="
