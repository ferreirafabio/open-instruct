#!/usr/bin/env python3
"""Compare original English and translated German samples side by side."""

import pandas as pd
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path("/work/dlclarge2/ferreira-oellm/open-instruct")
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer

def main():
    # Load original and translated data
    original_path = PROJECT_ROOT / "data/datasets_mixture_sft_preprocessed/argilla-distilabel-capybara-dpo-7k-binarized.parquet"
    translated_path = PROJECT_ROOT / "data/test_translation_chunked/de/argilla-distilabel-capybara-dpo-7k-binarized.parquet"

    print("=" * 80)
    print("ORIGINAL vs TRANSLATED COMPARISON")
    print("=" * 80)

    # Load tokenizer to count actual tokens
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

    # Max length used in translation (set to 150 in our test)
    max_length = 150
    chunk_threshold = max_length - 50  # 100 tokens triggers chunking

    print(f"Chunk threshold: {chunk_threshold} tokens (max_length={max_length} - 50)")
    print()

    # Load original (sampled same way as translation)
    import numpy as np
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(original_path)
    original_total = parquet_file.metadata.num_rows

    # Reproduce the same sampling (0.1% with seed 42)
    rng = np.random.default_rng(42)
    sample_fraction = 0.001
    n_samples = max(1, int(original_total * sample_fraction))
    sample_indices = np.sort(rng.choice(original_total, size=n_samples, replace=False))

    # Read only sampled rows
    chunk_size = 100000
    sampled_messages = []
    current_idx = 0
    sample_ptr = 0

    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=["messages"]):
        batch_end = current_idx + len(batch)
        while sample_ptr < len(sample_indices) and sample_indices[sample_ptr] < batch_end:
            local_idx = sample_indices[sample_ptr] - current_idx
            sampled_messages.append(batch["messages"][local_idx].as_py())
            sample_ptr += 1
        current_idx = batch_end
        if sample_ptr >= len(sample_indices):
            break

    original_df = pd.DataFrame({"messages": sampled_messages})
    translated_df = pd.read_parquet(translated_path)

    print(f"Original samples: {len(original_df)}")
    print(f"Translated samples: {len(translated_df)}")
    print()

    # Compare each sample
    chunked_count = 0
    for idx in range(min(len(original_df), len(translated_df))):
        print("=" * 80)
        print(f"SAMPLE {idx + 1}")
        print("=" * 80)

        orig_messages = original_df.iloc[idx]["messages"]
        trans_messages = translated_df.iloc[idx]["messages"]

        sample_had_chunking = False

        for msg_idx, (orig_msg, trans_msg) in enumerate(zip(orig_messages, trans_messages)):
            role = orig_msg['role'].upper()
            orig_content = orig_msg['content']
            trans_content = trans_msg['content']

            # Count tokens
            orig_tokens = len(tokenizer.encode(orig_content, add_special_tokens=False))
            trans_tokens = len(tokenizer.encode(trans_content, add_special_tokens=False))

            # Check if chunking would have been triggered
            would_chunk = orig_tokens > chunk_threshold
            if would_chunk:
                sample_had_chunking = True
                chunk_indicator = " ** CHUNKED **"
            else:
                chunk_indicator = ""

            print(f"\n--- {role} (msg {msg_idx + 1}) ---{chunk_indicator}")
            print(f"Original tokens: {orig_tokens}, Translated tokens: {trans_tokens}")

            if role == "SYSTEM":
                print("(System messages kept in English)")
                print(f"Content: {orig_content[:200]}..." if len(orig_content) > 200 else f"Content: {orig_content}")
            else:
                print()
                print("ORIGINAL (English):")
                print("-" * 40)
                if len(orig_content) > 500:
                    print(f"{orig_content[:500]}...")
                    print(f"... ({len(orig_content)} chars total)")
                else:
                    print(orig_content)

                print()
                print("TRANSLATED (German):")
                print("-" * 40)
                if len(trans_content) > 500:
                    print(f"{trans_content[:500]}...")
                    print(f"... ({len(trans_content)} chars total)")
                else:
                    print(trans_content)

        if sample_had_chunking:
            chunked_count += 1
            print("\n>>> This sample required CHUNKING <<<")

        print()

    print("=" * 80)
    print(f"SUMMARY: {chunked_count}/{len(original_df)} samples required chunking")
    print(f"(Chunk threshold: {chunk_threshold} tokens)")
    print("=" * 80)

if __name__ == "__main__":
    main()
