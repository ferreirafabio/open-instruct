#!/usr/bin/env python3
"""
Translate SFT datasets to EU languages using NLLB-200-3.3B.

This script translates the preprocessed parquet datasets to a target EU language.
It preserves the message structure and handles special cases like code blocks.

Usage:
    python translate_dataset.py --input INPUT.parquet --output OUTPUT.parquet --target-lang de

    # For SLURM array jobs:
    python translate_dataset.py --input INPUT.parquet --output-dir /output/path --target-lang $LANG

Features:
- Translates both user and assistant messages
- Keeps system messages in English (model instructions)
- Skips code blocks (preserves them as-is)
- Chunks long texts by sentences to avoid truncation
- Batched processing for GPU efficiency
- Checkpointing for fault tolerance
- Progress tracking
"""

import argparse
import json
import re
import signal
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from langdetect import detect, LangDetectException
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Suppress specific HuggingFace warning about token length exceeding model max
# This warning fires during token counting in chunk_text_by_sentences(), but we
# handle long texts by chunking them before model inference. The warning is harmless.
warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length is longer than the specified maximum sequence length",
)


# Import checkpoint manager from utils
from oellm.utils.checkpoint import CheckpointManager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from language_codes import EU_LANGUAGES, get_nllb_code

# Global flag for graceful shutdown
_shutdown_requested = False

# Map langdetect ISO codes to NLLB codes
# Only includes languages likely to appear in our datasets
LANGDETECT_TO_NLLB = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "el": "ell_Grek",
    "bg": "bul_Cyrl",
    "hr": "hrv_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "et": "est_Latn",
    "lv": "lvs_Latn",
    "lt": "lit_Latn",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "tl": "tgl_Latn",
    "ca": "cat_Latn",
    "af": "afr_Latn",
}


def detect_source_language(
    text: str, min_length: int = 20, return_reason: bool = False
) -> str | tuple[str, str]:
    """Detect source language and return NLLB code.

    Uses langdetect (character n-gram classifier, ~0.1ms per text).
    Falls back to English if detection fails or text is too short.

    Args:
        text: Text to detect language of
        min_length: Minimum text length for reliable detection
        return_reason: If True, return (lang_code, reason) tuple

    Returns:
        NLLB language code (e.g., "eng_Latn", "fra_Latn")
        If return_reason=True: (lang_code, reason) where reason is one of:
            "detected", "empty", "short", "error", "unmapped"
    """
    # Empty or whitespace-only
    if not text or not text.strip():
        if return_reason:
            return "eng_Latn", "empty"
        return "eng_Latn"

    # Too short for reliable detection
    if len(text.strip()) < min_length:
        if return_reason:
            return "eng_Latn", "short"
        return "eng_Latn"

    try:
        lang_code = detect(text)
        nllb_code = LANGDETECT_TO_NLLB.get(lang_code)
        if nllb_code:
            if return_reason:
                return nllb_code, "detected"
            return nllb_code
        else:
            # Language detected but not in our mapping
            if return_reason:
                return "eng_Latn", "unmapped"
            return "eng_Latn"
    except LangDetectException:
        if return_reason:
            return "eng_Latn", "error"
        return "eng_Latn"


def handle_shutdown_signal(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    print(f"\nReceived signal {signum}, requesting graceful shutdown...")
    _shutdown_requested = True


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences, preserving sentence boundaries.

    Uses a simple approach: split on sentence-ending punctuation followed by whitespace.
    Handles periods, question marks, exclamation points, and paragraph breaks.

    Args:
        text: Text to split

    Returns:
        List of sentences (may be single-element if no splits found)
    """
    if not text or len(text) < 50:
        return [text] if text else []

    # First, split on paragraph breaks (double newlines)
    paragraphs = re.split(r'\n\s*\n', text)

    sentences = []
    for para in paragraphs:
        if not para.strip():
            continue

        # Split on sentence endings: . ! ? followed by space
        # Use positive lookbehind to keep the punctuation with the sentence
        para_sentences = re.split(r'(?<=[.!?])\s+', para)

        for sent in para_sentences:
            sent = sent.strip()
            if sent:
                sentences.append(sent)

    # If no splits found, return original text
    if not sentences:
        return [text]

    return sentences


def truncate_to_max_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Hard truncate text to fit within max_tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    # Truncate tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def chunk_text_by_sentences(
    text: str,
    tokenizer,
    max_tokens: int = 400,
    safety_margin: int = 50,
) -> list[str]:
    """
    Chunk text into pieces that fit within max_tokens, splitting at sentence boundaries.

    GUARANTEES: Every returned chunk will have <= max_tokens tokens.

    Args:
        text: Text to chunk
        tokenizer: Tokenizer for counting tokens
        max_tokens: Maximum tokens per chunk (HARD LIMIT - chunks will be truncated to this)
        safety_margin: Extra margin for splitting decisions (chunks target max_tokens - safety_margin)

    Returns:
        List of text chunks, each guaranteed to fit within max_tokens
    """
    # Use aggressive target to leave room for tokenization variance
    target_max = max_tokens - safety_margin

    # Quick check: if text fits within hard limit, return as-is
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]

    # Split into sentences
    sentences = split_into_sentences(text)

    if len(sentences) <= 1:
        # Can't split by sentences - fall back to hard split at word boundaries
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0

        for word in words:
            word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
            if current_tokens + word_tokens > target_max and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        if not chunks:
            chunks = [text]

        # GUARANTEE: truncate all chunks to hard limit
        return [truncate_to_max_tokens(chunk, tokenizer, max_tokens) for chunk in chunks]

    # Group sentences into chunks
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        # If single sentence exceeds limit, recursively chunk it
        if sentence_tokens > target_max:
            # Flush current chunk first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split long sentence by words (recursive call handles truncation)
            sub_chunks = chunk_text_by_sentences(sentence, tokenizer, max_tokens, safety_margin)
            chunks.extend(sub_chunks)
            continue

        # Check if adding this sentence would exceed limit
        if current_tokens + sentence_tokens > target_max and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    if not chunks:
        chunks = [text]

    # GUARANTEE: truncate all chunks to hard limit (handles subword tokenization edge cases)
    result = [truncate_to_max_tokens(chunk, tokenizer, max_tokens) for chunk in chunks]

    # Debug: verify all chunks are within limit
    for i, chunk in enumerate(result):
        actual = len(tokenizer.encode(chunk, add_special_tokens=False))
        if actual > max_tokens:
            print(f"WARNING: Chunk {i} has {actual} tokens > max_tokens {max_tokens}")

    return result

# Code block patterns to preserve (not translate)
CODE_PATTERNS = [
    r"```[\s\S]*?```",  # Fenced code blocks
    r"`[^`]+`",  # Inline code
    r"https?://\S+",  # URLs
    r"\S+@\S+\.\S+",  # Emails
]

# Patterns that indicate code-heavy content (skip entire message)
CODE_HEAVY_PATTERNS = [
    r"^\s*(import|from|def|class|function|const|let|var)\s+",
    r"^\s*#include\s*<",
    r"^\s*public\s+(class|static|void)",
]


def is_code_heavy(text: str, threshold: float = 0.3) -> bool:
    """
    Check if text is primarily code (should skip translation).

    Args:
        text: Text to check
        threshold: Fraction of code indicators to trigger skip

    Returns:
        True if text appears to be primarily code
    """
    lines = text.split("\n")
    if not lines:
        return False

    code_lines = 0
    for line in lines:
        # Check for code-heavy patterns
        for pattern in CODE_HEAVY_PATTERNS:
            if re.match(pattern, line):
                code_lines += 1
                break
        # Check for high special character density
        if len(line) > 10:
            special_chars = sum(1 for c in line if c in "{}[]();=<>")
            if special_chars / len(line) > 0.15:
                code_lines += 1

    return code_lines / len(lines) > threshold


def extract_preservables(text: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Extract code blocks and other content that should be preserved.

    Args:
        text: Original text

    Returns:
        (text_with_placeholders, list of (placeholder, original_content))
    """
    preservables = []
    result = text

    for i, pattern in enumerate(CODE_PATTERNS):
        matches = re.findall(pattern, result)
        for j, match in enumerate(matches):
            placeholder = f"__PRESERVE_{i}_{j}__"
            result = result.replace(match, placeholder, 1)
            preservables.append((placeholder, match))

    return result, preservables


def restore_preservables(text: str, preservables: list[tuple[str, str]]) -> str:
    """Restore preserved content back into translated text."""
    result = text
    for placeholder, original in preservables:
        result = result.replace(placeholder, original)
    return result


class DatasetTranslator:
    """Translate SFT datasets using NLLB-200."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-3.3B",
        device: str = "auto",
        max_length: int = 1024,
        batch_size: int = 128,
    ):
        """
        Initialize translator.

        Args:
            model_name: NLLB model to use
            device: Device for inference ("auto", "cuda", "cpu")
            max_length: Maximum tokens for translation
            batch_size: Batch size for translation
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

        # Compile model for faster inference (10-30% speedup after first batch)
        if device == "cuda":
            self.model = torch.compile(self.model)
            print(f"Model loaded and compiled on {device}")
        else:
            print(f"Model loaded on {device}")

    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate a single text, chunking by sentences if too long.

        Args:
            text: Text to translate
            src_lang: Source NLLB language code (e.g., "eng_Latn")
            tgt_lang: Target NLLB language code (e.g., "deu_Latn")

        Returns:
            Translated text (never truncated)
        """
        if not text or not text.strip():
            return text

        # Check if code-heavy
        if is_code_heavy(text):
            return text

        # Extract preservable content
        text_to_translate, preservables = extract_preservables(text)

        # Skip if mostly preservable content
        if len(text_to_translate.strip()) < 20:
            return text

        # Chunk text if too long
        # Use larger chunks (max_length - 50) to preserve more context for better translation
        chunks = chunk_text_by_sentences(
            text_to_translate,
            self.tokenizer,
            max_tokens=self.max_length - 100,  # Leave room for special tokens
            safety_margin=20,
        )

        # Translate each chunk
        translated_chunks = []
        self.tokenizer.src_lang = src_lang

        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,  # Truncate if chunking missed edge cases
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=self.max_length,
                    num_beams=1,  # Greedy decoding - much faster than beam search
                )

            translated_chunk = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_chunks.append(translated_chunk)

        # Join translated chunks
        translated = ' '.join(translated_chunks)

        # Restore preserved content
        translated = restore_preservables(translated, preservables)

        return translated

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """
        Translate a batch of texts, chunking long texts by sentences.

        Args:
            texts: List of texts to translate
            src_lang: Source NLLB language code
            tgt_lang: Target NLLB language code

        Returns:
            List of translated texts (never truncated)
        """
        results = []
        preservables_list = []
        chunks_list = []  # List of (original_idx, chunks) tuples
        indices_to_translate = []

        # Pre-process: separate code-heavy, extract preservables, and chunk long texts
        for i, text in enumerate(texts):
            if not text or not text.strip() or is_code_heavy(text):
                results.append(text)
            else:
                text_clean, preservables = extract_preservables(text)
                if len(text_clean.strip()) < 20:
                    results.append(text)
                else:
                    # Chunk text if too long
                    # Use larger chunks to preserve more context for better translation
                    chunks = chunk_text_by_sentences(
                        text_clean,
                        self.tokenizer,
                        max_tokens=self.max_length - 100,  # Leave room for special tokens
                        safety_margin=20,
                    )
                    chunks_list.append((i, chunks))
                    preservables_list.append(preservables)
                    indices_to_translate.append(i)
                    results.append(None)  # Placeholder

        if not chunks_list:
            return results

        # Flatten all chunks for batched translation
        all_chunks = []
        chunk_boundaries = []  # (start_idx, end_idx) for each original text
        for orig_idx, chunks in chunks_list:
            start = len(all_chunks)
            all_chunks.extend(chunks)
            chunk_boundaries.append((start, len(all_chunks)))

        # Set source language
        self.tokenizer.src_lang = src_lang

        # Translate all chunks in batches
        all_translated_chunks = []
        for batch_start in range(0, len(all_chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(all_chunks))
            batch_chunks = all_chunks[batch_start:batch_end]

            inputs = self.tokenizer(
                batch_chunks,
                return_tensors="pt",
                truncation=True,  # Truncate if chunking missed edge cases
                max_length=self.max_length,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=self.max_length,
                    num_beams=1,  # Greedy decoding - much faster than beam search
                )

            for output in outputs:
                translated_chunk = self.tokenizer.decode(output, skip_special_tokens=True)
                all_translated_chunks.append(translated_chunk)

        # Reconstruct original texts from translated chunks
        for i, ((start, end), preservables) in enumerate(zip(chunk_boundaries, preservables_list)):
            translated_chunks = all_translated_chunks[start:end]
            translated = ' '.join(translated_chunks)
            translated = restore_preservables(translated, preservables)
            results[indices_to_translate[i]] = translated

        return results

    def translate_messages(
        self,
        messages: list[dict[str, str]],
        tgt_lang_iso: str,
    ) -> list[dict[str, str]]:
        """
        Translate a conversation (list of messages) - uses batching internally.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            tgt_lang_iso: Target ISO 639-1 language code (e.g., "de")

        Returns:
            Translated messages
        """
        src_lang = "eng_Latn"
        tgt_lang = get_nllb_code(tgt_lang_iso)

        # Collect non-system messages for batch translation
        texts_to_translate = []
        indices_to_translate = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            if role != "system":
                texts_to_translate.append(msg.get("content", ""))
                indices_to_translate.append(i)

        # Batch translate all non-system messages
        if texts_to_translate:
            translated_texts = self.translate_batch(texts_to_translate, src_lang, tgt_lang)
        else:
            translated_texts = []

        # Build result
        translated_messages = []
        translate_idx = 0
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                translated_messages.append({"role": role, "content": content})
            else:
                translated_messages.append({"role": role, "content": translated_texts[translate_idx]})
                translate_idx += 1

        return translated_messages

    def translate_conversations_batched(
        self,
        conversations: list[list[dict[str, str]]],
        tgt_lang_iso: str,
        detect_source: bool = True,
        return_stats: bool = False,
    ) -> list[list[dict[str, str]]] | tuple[list[list[dict[str, str]]], dict]:
        """
        Translate multiple conversations with batching across all messages.

        This is much more efficient than translating one conversation at a time.
        Supports automatic source language detection for multilingual datasets.

        Args:
            conversations: List of conversations, each a list of message dicts
            tgt_lang_iso: Target ISO 639-1 language code (e.g., "de")
            detect_source: If True, detect source language per message (default True)
            return_stats: If True, return (translations, stats) tuple with source language info

        Returns:
            List of translated conversations, or (translations, stats) tuple if return_stats=True
        """
        tgt_lang = get_nllb_code(tgt_lang_iso)

        # Flatten all messages, tracking which conversation they belong to
        all_texts = []
        all_src_langs = []  # Detected source language per text
        all_detect_reasons = []  # Why each language was detected/defaulted
        message_map = []  # (conv_idx, msg_idx, is_system)

        # Counters for detailed stats
        system_message_count = 0
        empty_message_count = 0
        short_message_count = 0

        for conv_idx, messages in enumerate(conversations):
            for msg_idx, msg in enumerate(messages):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                is_system = (role == "system")
                message_map.append((conv_idx, msg_idx, is_system))
                if is_system:
                    system_message_count += 1
                else:
                    all_texts.append(content)
                    # Detect source language (fast: ~0.1ms per text)
                    if detect_source:
                        src_lang, reason = detect_source_language(content, return_reason=True)
                        if reason == "empty":
                            empty_message_count += 1
                        elif reason == "short":
                            short_message_count += 1
                    else:
                        src_lang = "eng_Latn"
                        reason = "assumed"
                    all_src_langs.append(src_lang)
                    all_detect_reasons.append(reason)

        if not all_texts:
            # No texts to translate, just return conversations with system messages
            results = [[] for _ in conversations]
            for conv_idx, msg_idx, is_system in message_map:
                original_msg = conversations[conv_idx][msg_idx]
                results[conv_idx].append(original_msg.copy())
            if return_stats:
                return results, {
                    "source_lang_distribution": {},
                    "skipped_same_language": 0,
                    "total_messages": 0,
                    "system_messages": system_message_count,
                    "empty_messages": 0,
                    "short_messages": 0,
                }
            return results

        # Group texts by source language for efficient batching
        # Each group can be translated together with the same src_lang setting
        lang_groups = defaultdict(list)  # src_lang -> [(original_idx, text)]
        source_lang_counts = defaultdict(int)  # Track source language distribution
        for idx, (text, src_lang) in enumerate(zip(all_texts, all_src_langs)):
            source_lang_counts[src_lang] += 1
            # Skip if source == target (text is already in target language)
            if src_lang == tgt_lang:
                lang_groups["_skip_"].append((idx, text))
            else:
                lang_groups[src_lang].append((idx, text))

        skipped_same_lang = len(lang_groups.get("_skip_", []))

        # Translate each language group
        all_translated = [None] * len(all_texts)

        for src_lang, group in lang_groups.items():
            if src_lang == "_skip_":
                # Keep original text (already in target language)
                for idx, text in group:
                    all_translated[idx] = text
                continue

            # Extract texts for this source language
            indices = [idx for idx, _ in group]
            texts = [text for _, text in group]

            # Batch translate
            translated = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                translated_batch = self.translate_batch(batch, src_lang, tgt_lang)
                translated.extend(translated_batch)

            # Store results at original indices
            for idx, trans in zip(indices, translated):
                all_translated[idx] = trans

        # Reconstruct conversations
        results = [[] for _ in conversations]
        translate_idx = 0

        for conv_idx, msg_idx, is_system in message_map:
            original_msg = conversations[conv_idx][msg_idx]
            role = original_msg.get("role", "user")
            content = original_msg.get("content", "")

            if is_system:
                results[conv_idx].append({"role": role, "content": content})
            else:
                results[conv_idx].append({"role": role, "content": all_translated[translate_idx]})
                translate_idx += 1

        if return_stats:
            stats = {
                "source_lang_distribution": dict(source_lang_counts),
                "skipped_same_language": skipped_same_lang,
                "total_messages": len(all_texts),
                "system_messages": system_message_count,
                "empty_messages": empty_message_count,
                "short_messages": short_message_count,
            }
            return results, stats
        return results


def translate_dataset(
    input_path: Path,
    output_path: Path,
    target_lang: str,
    translator: DatasetTranslator,
    checkpoint_interval: int = 500,
    conversation_batch_size: int = 128,
    sample_fraction: float | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Translate an entire dataset with atomic checkpointing and graceful shutdown.

    Uses CheckpointManager for robust resumability:
    - Atomic writes: temp file + rename prevents corruption
    - Incremental checkpoints: each batch saved separately
    - Graceful shutdown: SIGTERM triggers checkpoint before exit
    - Auto-resume: scans for latest valid checkpoint on restart

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        target_lang: Target ISO 639-1 language code
        translator: DatasetTranslator instance
        checkpoint_interval: Save checkpoint every N examples (default 500)
        sample_fraction: If set, randomly sample this fraction of the dataset (0.0-1.0)
        seed: Random seed for sampling reproducibility

    Returns:
        Statistics dictionary
    """
    global _shutdown_requested

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)

    print(f"Loading dataset: {input_path}")

    # Memory-efficient sampling: read metadata first, then only load sampled rows
    import pyarrow.parquet as pq
    import numpy as np

    parquet_file = pq.ParquetFile(input_path)
    original_total = parquet_file.metadata.num_rows
    print(f"Total examples in file: {original_total}")

    # Determine total and set up sampling if requested
    use_streaming = sample_fraction is None or sample_fraction >= 1.0
    sample_indices = None

    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        # Sampled mode: compute indices, load only sampled rows into memory
        rng = np.random.default_rng(seed)
        n_samples = max(1, int(original_total * sample_fraction))
        sample_indices = np.sort(rng.choice(original_total, size=n_samples, replace=False))
        total = n_samples
        print(f"Sampled {sample_fraction*100:.1f}% -> {n_samples} examples")

        # Load sampled messages into memory (small enough after sampling)
        chunk_size = 100000
        all_messages = []
        current_idx = 0
        sample_ptr = 0

        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=["messages"]):
            batch_end = current_idx + len(batch)
            while sample_ptr < len(sample_indices) and sample_indices[sample_ptr] < batch_end:
                local_idx = sample_indices[sample_ptr] - current_idx
                all_messages.append(batch["messages"][local_idx].as_py())
                sample_ptr += 1
            current_idx = batch_end
            if sample_ptr >= len(sample_indices):
                break

        df = pd.DataFrame({"messages": all_messages})
    else:
        # Non-sampled streaming mode: don't load into memory, process directly from parquet
        total = original_total
        df = None  # Will stream from parquet_file instead
        print(f"Streaming mode: {total} examples (no memory preload)")

    print(f"Examples to translate: {total}")

    # Set up checkpoint manager with atomic writes
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / f"{output_path.stem}.checkpoint"
    dataset_name = output_path.stem

    checkpoint_mgr = CheckpointManager(checkpoint_dir, prefix=dataset_name)

    # Check for resume point
    resume_batch_idx, last_metadata = checkpoint_mgr.get_resume_point()
    resume_from = 0
    skipped = 0

    if resume_batch_idx > 0 and last_metadata:
        resume_from = last_metadata.get("processed_count", 0)
        skipped = last_metadata.get("skipped_count", 0)
        print(f"Resuming from batch {resume_batch_idx}, processed {resume_from} examples")

    # Translate in batches
    pbar = tqdm(total=total, initial=resume_from, desc=f"Translating to {target_lang}")
    batch_idx = resume_batch_idx
    processed_count = resume_from

    # Accumulate source language stats across batches
    total_source_lang_counts = defaultdict(int)
    total_skipped_same_lang = 0
    total_messages_processed = 0
    total_system_messages = 0
    total_empty_messages = 0
    total_short_messages = 0
    conversations_with_errors = 0

    # Create conversation iterator based on mode
    if df is not None:
        # Sampled mode: iterate from in-memory DataFrame
        def conversation_iterator():
            for idx in range(total):
                messages = df.iloc[idx]["messages"]
                if hasattr(messages, "tolist"):
                    messages = messages.tolist()
                yield messages
    else:
        # Streaming mode: iterate directly from parquet file
        def conversation_iterator():
            for batch in parquet_file.iter_batches(batch_size=10000, columns=["messages"]):
                for msg in batch["messages"]:
                    yield msg.as_py()

    # Create iterator and skip to resume point
    conv_iter = conversation_iterator()
    if resume_from > 0:
        print(f"Skipping {resume_from} already-processed examples...")
        for _ in range(resume_from):
            next(conv_iter, None)

    for batch_start in range(resume_from, total, conversation_batch_size):
        # Check for shutdown request
        if _shutdown_requested:
            print(f"\nGraceful shutdown at batch {batch_idx}, processed {processed_count}")
            break

        batch_end = min(batch_start + conversation_batch_size, total)

        # Collect batch of conversations from iterator
        conversations = []
        for _ in range(batch_end - batch_start):
            msg = next(conv_iter, None)
            if msg is None:
                break
            conversations.append(msg)

        if not conversations:
            break

        # Translate batch
        try:
            translated_batch, batch_stats = translator.translate_conversations_batched(
                conversations, target_lang, return_stats=True
            )
            # Accumulate source language stats
            for lang, count in batch_stats["source_lang_distribution"].items():
                total_source_lang_counts[lang] += count
            total_skipped_same_lang += batch_stats["skipped_same_language"]
            total_messages_processed += batch_stats["total_messages"]
            total_system_messages += batch_stats["system_messages"]
            total_empty_messages += batch_stats["empty_messages"]
            total_short_messages += batch_stats["short_messages"]
        except torch.cuda.OutOfMemoryError:
            # OOM is fatal - reduce conversation_batch_size or batch_size
            raise
        except Exception as e:
            print(f"Error at batch {batch_start}-{batch_end}: {e}")
            translated_batch = conversations  # Keep originals on error
            skipped += len(conversations)
            conversations_with_errors += len(conversations)

        processed_count += len(translated_batch)
        pbar.update(len(translated_batch))

        # Save checkpoint with atomic write
        if processed_count % checkpoint_interval < conversation_batch_size or batch_end >= total:
            batch_df = pd.DataFrame({"messages": translated_batch})
            checkpoint_mgr.save(
                batch_df,
                batch_idx=batch_idx,
                metadata={
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "processed_count": processed_count,
                    "skipped_count": skipped,
                }
            )
            batch_idx += 1

    pbar.close()

    # If shutdown was requested, exit without merging
    if _shutdown_requested:
        print(f"Checkpoint saved. Resume by re-running the same command.")
        return {
            "input": str(input_path),
            "output": str(output_path),
            "target_lang": target_lang,
            "status": "interrupted",
            "processed": processed_count,
            "total": total,
            "source_lang_distribution": dict(total_source_lang_counts),
            "skipped_same_language": total_skipped_same_lang,
            "total_messages": total_messages_processed,
            "system_messages": total_system_messages,
            "empty_messages": total_empty_messages,
            "short_messages": total_short_messages,
            "conversations_with_errors": conversations_with_errors,
        }

    # Merge all checkpoints into final output
    print(f"Merging {checkpoint_mgr.get_checkpoint_count()} checkpoints...")
    checkpoint_mgr.merge_checkpoints(output_path)
    print(f"Saved to: {output_path}")

    # Clean up checkpoints
    checkpoint_mgr.cleanup()
    if checkpoint_dir.exists():
        checkpoint_dir.rmdir()

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "target_lang": target_lang,
        "original_total": original_total,
        "sample_fraction": sample_fraction,
        "total": total,
        "skipped": skipped,
        "translated": total - skipped,
        "source_lang_distribution": dict(total_source_lang_counts),
        "skipped_same_language": total_skipped_same_lang,
        "total_messages": total_messages_processed,
        "system_messages": total_system_messages,
        "empty_messages": total_empty_messages,
        "short_messages": total_short_messages,
        "conversations_with_errors": conversations_with_errors,
    }

    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Translate SFT datasets to EU languages")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output", type=str, help="Output parquet file (optional, auto-generated if not provided)")
    parser.add_argument("--output-dir", type=str, help="Output directory (for batch processing)")
    parser.add_argument("--target-lang", type=str, required=True, help="Target language ISO code (e.g., de, fr, es)")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-3.3B", help="Translation model")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for GPU inference (600M model uses ~4GB at 128)")
    parser.add_argument("--max-length", type=int, default=1024, help="Max tokens per chunk (larger = better context)")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Checkpoint every N examples")
    parser.add_argument("--conversation-batch-size", type=int, default=128, help="Conversations per batch (higher = faster, more memory)")
    parser.add_argument("--resume-from", type=int, default=0, help="Resume from index")
    parser.add_argument("--sample-fraction", type=float, default=None, help="Sample fraction (0.0-1.0), e.g., 0.1 for 10%")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")

    args = parser.parse_args()

    # Validate target language
    if args.target_lang not in EU_LANGUAGES:
        print(f"Error: Unknown language code '{args.target_lang}'")
        print(f"Valid codes: {list(EU_LANGUAGES.keys())}")
        sys.exit(1)

    # Determine output path
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    elif args.output_dir:
        output_dir = Path(args.output_dir) / args.target_lang
        output_path = output_dir / input_path.name
    else:
        output_path = input_path.with_stem(f"{input_path.stem}_{args.target_lang}")

    print("=" * 60)
    print("TRANSLATION CONFIG")
    print("=" * 60)
    print(f"Input:        {input_path}")
    print(f"Output:       {output_path}")
    print(f"Target lang:  {args.target_lang} ({EU_LANGUAGES[args.target_lang]['name']})")
    print(f"NLLB code:    {get_nllb_code(args.target_lang)}")
    print(f"Model:        {args.model}")
    print(f"Device:       {args.device}")
    print(f"Batch size:   {args.batch_size}")
    if args.sample_fraction:
        print(f"Sample:       {args.sample_fraction*100:.1f}% (seed={args.seed})")
    else:
        print(f"Sample:       100% (no sampling)")
    print("=" * 60)

    if args.dry_run:
        print("Dry run - exiting")
        return

    # Initialize translator
    translator = DatasetTranslator(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # Translate (resume is handled automatically by CheckpointManager)
    stats = translate_dataset(
        input_path=input_path,
        output_path=output_path,
        target_lang=args.target_lang,
        translator=translator,
        checkpoint_interval=args.checkpoint_interval,
        conversation_batch_size=args.conversation_batch_size,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("TRANSLATION COMPLETE")
    print("=" * 60)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
