#!/usr/bin/env python3
"""Debug script to visualize chunking and merging in translation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path("/work/dlclarge2/ferreira-oellm/open-instruct")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "oellm/pipelines/translation"))

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from translate_dataset import chunk_text_by_sentences, extract_preservables, restore_preservables
from language_codes import get_nllb_code

def main():
    print("=" * 80)
    print("CHUNKING DEBUG: Showing chunks before/after translation")
    print("=" * 80)

    # Load tokenizer and model
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Load original data (same sampling as test)
    original_path = PROJECT_ROOT / "data/datasets_mixture_sft_preprocessed/argilla-distilabel-capybara-dpo-7k-binarized.parquet"

    parquet_file = pq.ParquetFile(original_path)
    original_total = parquet_file.metadata.num_rows

    # Same sampling as test (0.1% with seed 42)
    rng = np.random.default_rng(42)
    sample_fraction = 0.001
    n_samples = max(1, int(original_total * sample_fraction))
    sample_indices = np.sort(rng.choice(original_total, size=n_samples, replace=False))

    # Read sampled rows
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

    print(f"\nLoaded {len(sampled_messages)} samples")

    # Use forced chunking threshold (same as test: max_length=150)
    max_length = 150
    chunk_threshold = max_length - 50  # 100 tokens

    print(f"Chunk threshold: {chunk_threshold} tokens (max_length={max_length})")

    # Process first 2 samples to show chunking in detail
    src_lang = "eng_Latn"
    tgt_lang = get_nllb_code("de")

    # Just show 1 sample, 2 messages for cleaner output with full text
    for sample_idx in range(min(1, len(sampled_messages))):
        messages = sampled_messages[sample_idx]

        print("\n" + "=" * 80)
        print(f"SAMPLE {sample_idx + 1}")
        print("=" * 80)

        for msg_idx, msg in enumerate(messages[:2]):  # First 2 messages (user + assistant)
            role = msg['role'].upper()
            content = msg['content']

            if role == "SYSTEM":
                print(f"\n--- {role} (kept in English) ---")
                print(content[:200] + "..." if len(content) > 200 else content)
                continue

            # Count tokens
            tokens = tokenizer.encode(content, add_special_tokens=False)
            token_count = len(tokens)

            print(f"\n{'='*60}")
            print(f"MESSAGE: {role} (msg {msg_idx + 1})")
            print(f"{'='*60}")
            print(f"Token count: {token_count} (threshold: {chunk_threshold})")
            print(f"Will chunk: {'YES' if token_count > chunk_threshold else 'NO'}")

            # Show original English
            print(f"\n>>> ORIGINAL ENGLISH ({len(content)} chars):")
            print("-" * 40)
            print(content)
            print("-" * 40)

            # Extract preservables (code blocks, URLs)
            text_clean, preservables = extract_preservables(content)
            if preservables:
                print(f"\n>>> PRESERVED (not translated): {len(preservables)} items")
                for placeholder, original in preservables[:3]:
                    print(f"  - {placeholder}: {original[:50]}...")

            # Chunk the text
            chunks = chunk_text_by_sentences(
                text_clean,
                tokenizer,
                max_tokens=max_length - 50,
                safety_margin=20,
            )

            print(f"\n>>> CHUNKS CREATED: {len(chunks)}")
            print("-" * 40)

            # Translate each chunk and show it
            translated_chunks = []
            tokenizer.src_lang = src_lang

            for i, chunk in enumerate(chunks):
                chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
                print(f"\nCHUNK {i+1}/{len(chunks)} ({chunk_tokens} tokens):")
                print(f"  EN: {chunk}")

                # Translate this chunk
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=False,
                    max_length=max_length,
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True,
                    )

                translated_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated_chunks.append(translated_chunk)
                print(f"  DE: {translated_chunk}")

            # Merge chunks
            merged = ' '.join(translated_chunks)

            # Restore preserved content
            final = restore_preservables(merged, preservables)

            print(f"\n>>> MERGED GERMAN OUTPUT ({len(final)} chars):")
            print("-" * 40)
            print(final)
            print("-" * 40)

            print(f"\n>>> SUMMARY:")
            print(f"  Original: {len(content)} chars, {token_count} tokens")
            print(f"  Chunks: {len(chunks)}")
            print(f"  Final: {len(final)} chars")

if __name__ == "__main__":
    main()
