"""Tests for Dolci language distribution analysis."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from oellm.experiments.dolci_distribution.scripts.detect_language_distribution import (
    aggregate_results,
    cmd_merge,
    detect_languages_batch,
    extract_user_text,
    merge_shard_results,
)


# --- Helpers ---


def make_messages(user_texts: list[str], assistant_texts: list[str] | None = None) -> list[dict]:
    """Build a messages list from user (and optional assistant) texts."""
    msgs = []
    if assistant_texts is None:
        assistant_texts = ["OK"] * len(user_texts)
    for u, a in zip(user_texts, assistant_texts):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def make_dataset(
    texts: list[str],
    lang: str = "en",
    source_dataset: str = "test",
    domain: str = "general",
) -> Dataset:
    """Create a minimal HF Dataset with messages and metadata."""
    records = []
    for i, text in enumerate(texts):
        records.append(
            {
                "messages": make_messages([text]),
                "source_dataset": source_dataset,
                "domain": domain,
                "id": f"{lang}_{i}",
            }
        )
    return Dataset.from_list(records)


# --- extract_user_text tests ---


class TestExtractUserText:
    def test_basic(self):
        example = {"messages": make_messages(["Hello world", "How are you?"])}
        result = extract_user_text(example)
        assert result["user_text"] == "Hello world How are you?"

    def test_skips_assistant(self):
        example = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "I am an AI"},
                {"role": "user", "content": "Thanks"},
            ]
        }
        result = extract_user_text(example)
        assert "I am an AI" not in result["user_text"]
        assert result["user_text"] == "Hello Thanks"

    def test_empty_content(self):
        example = {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "user", "content": None},
                {"role": "user", "content": "Valid text"},
            ]
        }
        result = extract_user_text(example)
        assert result["user_text"] == "Valid text"

    def test_no_user_messages(self):
        example = {"messages": [{"role": "assistant", "content": "Hello"}]}
        result = extract_user_text(example)
        assert result["user_text"] == ""


# --- detect_languages_batch tests ---


class TestDetectLanguagesBatch:
    def test_english(self):
        batch = {
            "user_text": [
                "This is a fairly long English sentence that should be detected correctly."
            ]
        }
        result = detect_languages_batch(batch)
        assert result["detected_lang"][0] == "en"
        assert 0.0 < result["detection_score"][0] <= 1.0

    def test_german(self):
        batch = {
            "user_text": [
                "Dies ist ein ziemlich langer deutscher Satz, der korrekt erkannt werden sollte."
            ]
        }
        result = detect_languages_batch(batch)
        assert result["detected_lang"][0] == "de"

    def test_french(self):
        batch = {
            "user_text": [
                "Ceci est une phrase assez longue en français qui devrait être détectée correctement."
            ]
        }
        result = detect_languages_batch(batch)
        assert result["detected_lang"][0] == "fr"

    def test_scores_in_range(self):
        batch = {
            "user_text": [
                "Hello world, this is a test.",
                "Hallo Welt, dies ist ein Test.",
            ]
        }
        result = detect_languages_batch(batch)
        for score in result["detection_score"]:
            assert 0.0 <= score <= 1.0

    def test_short_text_returns_unknown(self):
        batch = {"user_text": ["Hi", "OK", "x"]}
        result = detect_languages_batch(batch)
        for lang, score in zip(result["detected_lang"], result["detection_score"]):
            assert lang == "unknown"
            assert score == 0.0

    def test_empty_text(self):
        batch = {"user_text": ["", "   "]}
        result = detect_languages_batch(batch)
        assert result["detected_lang"] == ["unknown", "unknown"]
        assert result["detection_score"] == [0.0, 0.0]


# --- Math/code content tests ---


class TestMathCodeContent:
    def test_mixed_math_text(self):
        """English text with math should still be detected as English."""
        batch = {
            "user_text": [
                "Please solve the following equation for x: 2x + 3 = 7. "
                "Show your work step by step and explain each step."
            ]
        }
        result = detect_languages_batch(batch)
        assert result["detected_lang"][0] == "en"
        assert result["detection_score"][0] > 0.3

    def test_pure_math_low_confidence(self):
        """Pure math expressions should have low confidence or be unknown."""
        batch = {"user_text": ["2x + 3 = 7; f(x) = x^2 + 3x - 5; lim x->0 sin(x)/x = 1"]}
        result = detect_languages_batch(batch)
        # Either low confidence or unknown - math isn't really any language
        assert result["detection_score"][0] < 0.8 or result["detected_lang"][0] == "unknown"

    def test_code_text(self):
        """Code snippets may detect as English or have lower confidence."""
        batch = {
            "user_text": [
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    "
                "return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(10))"
            ]
        }
        result = detect_languages_batch(batch)
        # Code is usually classified as some language - just check it doesn't crash
        assert result["detected_lang"][0] != ""
        assert 0.0 <= result["detection_score"][0] <= 1.0


# --- aggregate_results tests ---


class TestAggregateResults:
    def _make_detected_dataset(self, langs, scores, sources=None, domains=None):
        """Create a dataset that looks like post-detection output."""
        n = len(langs)
        data = {
            "user_text": ["test text " * 10] * n,
            "detected_lang": langs,
            "detection_score": scores,
        }
        if sources is not None:
            data["source_dataset"] = sources
        if domains is not None:
            data["domain"] = domains
        return Dataset.from_dict(data)

    def test_basic_aggregation(self):
        ds = self._make_detected_dataset(
            langs=["en", "en", "de", "fr", "en"],
            scores=[0.99, 0.95, 0.88, 0.92, 0.97],
        )
        result = aggregate_results(ds)
        assert result["total_samples"] == 5
        assert result["language_counts"]["en"] == 3
        assert result["language_counts"]["de"] == 1
        assert result["language_counts"]["fr"] == 1
        assert result["language_percentages"]["en"] == 60.0

    def test_confidence_statistics(self):
        scores = [0.99, 0.3, 0.88, 0.45, 0.97]
        ds = self._make_detected_dataset(
            langs=["en", "en", "de", "fr", "en"],
            scores=scores,
        )
        result = aggregate_results(ds)
        stats = result["confidence_stats"]
        assert abs(stats["mean"] - np.mean(scores)) < 1e-6
        assert abs(stats["median"] - np.median(scores)) < 1e-6
        assert stats["low_confidence_count"] == 2  # 0.3 and 0.45

    def test_per_source_breakdown(self):
        ds = self._make_detected_dataset(
            langs=["en", "en", "de", "de"],
            scores=[0.9, 0.9, 0.9, 0.9],
            sources=["src_a", "src_a", "src_b", "src_b"],
        )
        result = aggregate_results(ds)
        assert result["per_source_dataset"]["src_a"] == {"en": 2}
        assert result["per_source_dataset"]["src_b"] == {"de": 2}

    def test_per_domain_breakdown(self):
        ds = self._make_detected_dataset(
            langs=["en", "fr", "en", "de"],
            scores=[0.9, 0.9, 0.9, 0.9],
            domains=["math", "math", "code", "code"],
        )
        result = aggregate_results(ds)
        assert result["per_domain"]["math"]["en"] == 1
        assert result["per_domain"]["math"]["fr"] == 1
        assert result["per_domain"]["code"]["en"] == 1
        assert result["per_domain"]["code"]["de"] == 1

    def test_low_confidence_threshold(self):
        ds = self._make_detected_dataset(
            langs=["en", "de", "fr", "zh", "ja"],
            scores=[0.99, 0.49, 0.01, 0.50, 0.3],
        )
        result = aggregate_results(ds)
        low = result["confidence_stats"]["low_confidence_languages"]
        # 0.49, 0.01, 0.3 are < 0.5
        assert result["confidence_stats"]["low_confidence_count"] == 3
        assert "de" in low
        assert "fr" in low
        assert "ja" in low
        assert "en" not in low


# --- Parallelization tests ---


class TestParallelization:
    @pytest.fixture()
    def small_dataset(self):
        """Create a small dataset for parallel testing."""
        texts_and_langs = [
            ("This is an English text that should be long enough for detection.", "en"),
            (
                "Dies ist ein deutscher Text, der lang genug für die Erkennung sein sollte.",
                "de",
            ),
            (
                "Ceci est un texte français qui devrait être assez long pour la détection.",
                "fr",
            ),
            (
                "Este es un texto en español que debería ser lo suficientemente largo.",
                "es",
            ),
        ] * 10  # 40 samples

        records = []
        for i, (text, _) in enumerate(texts_and_langs):
            records.append(
                {
                    "messages": make_messages([text]),
                    "source_dataset": "test",
                    "domain": "general",
                    "id": str(i),
                }
            )
        return Dataset.from_list(records)

    def test_parallel_vs_sequential(self, small_dataset):
        ds1 = small_dataset.map(extract_user_text, num_proc=1)
        ds1 = ds1.map(detect_languages_batch, batched=True, batch_size=10, num_proc=1)

        ds2 = small_dataset.map(extract_user_text, num_proc=2)
        ds2 = ds2.map(detect_languages_batch, batched=True, batch_size=10, num_proc=2)

        assert ds1["detected_lang"] == ds2["detected_lang"]
        assert ds1["detection_score"] == ds2["detection_score"]

    def test_parallel_preserves_order(self, small_dataset):
        ds = small_dataset.map(extract_user_text, num_proc=2)
        ds = ds.map(detect_languages_batch, batched=True, batch_size=10, num_proc=2)
        # IDs should still be in order
        assert ds["id"] == [str(i) for i in range(40)]

    def test_batched_vs_unbatched(self, small_dataset):
        ds_base = small_dataset.map(extract_user_text, num_proc=1)

        # Batched
        ds_batched = ds_base.map(
            detect_languages_batch, batched=True, batch_size=5, num_proc=1
        )

        # "Unbatched" = batch_size=1
        ds_unbatched = ds_base.map(
            detect_languages_batch, batched=True, batch_size=1, num_proc=1
        )

        assert ds_batched["detected_lang"] == ds_unbatched["detected_lang"]


# --- Integration test ---


class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        """Run the full pipeline on a small mock dataset."""
        records = [
            {
                "messages": make_messages(
                    ["Tell me about the history of France and its culture in detail."]
                ),
                "source_dataset": "wiki",
                "domain": "general",
                "id": "en_0",
            },
            {
                "messages": make_messages(
                    [
                        "Erzählen Sie mir über die Geschichte Deutschlands und seine Kultur im Detail."
                    ]
                ),
                "source_dataset": "wiki",
                "domain": "general",
                "id": "de_0",
            },
            {
                "messages": make_messages(
                    [
                        "Parlez-moi de l'histoire de la France et de sa culture en détail s'il vous plaît."
                    ]
                ),
                "source_dataset": "translated",
                "domain": "general",
                "id": "fr_0",
            },
        ]
        ds = Dataset.from_list(records)

        # Run pipeline
        ds = ds.map(extract_user_text, num_proc=1)
        ds = ds.map(detect_languages_batch, batched=True, batch_size=10, num_proc=1)
        results = aggregate_results(ds)

        # Check structure
        assert results["total_samples"] == 3
        assert "en" in results["language_counts"]
        assert "de" in results["language_counts"]
        assert "fr" in results["language_counts"]
        assert "confidence_stats" in results
        assert "per_source_dataset" in results
        assert "per_domain" in results

        # Check JSON serializable
        output_file = tmp_path / "test_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        assert output_file.exists()

        # Reload and verify
        with open(output_file) as f:
            loaded = json.load(f)
        assert loaded["total_samples"] == 3


# --- Helpers for shard merge tests ---


def make_shard_result(lang_counts, scores, source_counts=None, domain_counts=None):
    """Build a shard result dict with known values for merge testing."""
    total = sum(lang_counts.values())
    return {
        "total_samples": total,
        "language_counts": lang_counts,
        "language_percentages": {l: round(c / total * 100, 4) for l, c in lang_counts.items()},
        "confidence_stats": {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(min(scores)),
            "max": float(max(scores)),
            "low_confidence_count": sum(1 for s in scores if s < 0.5),
            "low_confidence_percentage": round(
                sum(1 for s in scores if s < 0.5) / len(scores) * 100, 4
            ),
            "low_confidence_languages": {},
        },
        "text_length_stats": {"mean": 100.0, "median": 90.0, "std": 20.0},
        "per_source_dataset": source_counts or {},
        "per_domain": domain_counts or {},
    }


# --- Merge shard results tests ---


class TestMergeShardResults:
    def test_merge_two_shards(self):
        """Merge two shards and verify total_samples, language_counts, and percentages."""
        shard1 = make_shard_result({"en": 6, "de": 4}, [0.9, 0.8, 0.95, 0.7, 0.85, 0.9, 0.75, 0.88, 0.92, 0.6])
        shard2 = make_shard_result({"en": 3, "fr": 7}, [0.85, 0.9, 0.7, 0.6, 0.95, 0.88, 0.92, 0.75, 0.8, 0.91])

        merged = merge_shard_results([shard1, shard2])

        assert merged["total_samples"] == 20
        assert merged["language_counts"]["en"] == 9
        assert merged["language_counts"]["de"] == 4
        assert merged["language_counts"]["fr"] == 7
        # Percentages should be recalculated from merged totals
        assert merged["language_percentages"]["en"] == round(9 / 20 * 100, 4)
        assert merged["language_percentages"]["de"] == round(4 / 20 * 100, 4)
        assert merged["language_percentages"]["fr"] == round(7 / 20 * 100, 4)

    def test_merge_per_source_dataset(self):
        """Two shards with overlapping source_datasets should have counts merged."""
        shard1 = make_shard_result(
            {"en": 5},
            [0.9] * 5,
            source_counts={"wiki": {"en": 3}, "books": {"en": 2}},
        )
        shard2 = make_shard_result(
            {"en": 4, "de": 1},
            [0.9] * 5,
            source_counts={"wiki": {"en": 2, "de": 1}, "news": {"en": 2}},
        )

        merged = merge_shard_results([shard1, shard2])

        assert merged["per_source_dataset"]["wiki"]["en"] == 5
        assert merged["per_source_dataset"]["wiki"]["de"] == 1
        assert merged["per_source_dataset"]["books"]["en"] == 2
        assert merged["per_source_dataset"]["news"]["en"] == 2

    def test_merge_per_domain(self):
        """Two shards with overlapping domains should have counts merged."""
        shard1 = make_shard_result(
            {"en": 3, "fr": 2},
            [0.9] * 5,
            domain_counts={"math": {"en": 2, "fr": 1}, "code": {"en": 1, "fr": 1}},
        )
        shard2 = make_shard_result(
            {"en": 2, "fr": 3},
            [0.9] * 5,
            domain_counts={"math": {"en": 1, "fr": 2}, "general": {"en": 1, "fr": 1}},
        )

        merged = merge_shard_results([shard1, shard2])

        assert merged["per_domain"]["math"]["en"] == 3
        assert merged["per_domain"]["math"]["fr"] == 3
        assert merged["per_domain"]["code"]["en"] == 1
        assert merged["per_domain"]["code"]["fr"] == 1
        assert merged["per_domain"]["general"]["en"] == 1
        assert merged["per_domain"]["general"]["fr"] == 1

    def test_merge_confidence_stats(self):
        """Verify weighted mean is correct, min/max are global extremes, low_confidence_count is sum."""
        shard1 = make_shard_result({"en": 4}, [0.9, 0.8, 0.3, 0.95])  # 1 low conf
        shard2 = make_shard_result({"en": 6}, [0.4, 0.2, 0.85, 0.7, 0.6, 0.45])  # 3 low conf

        merged = merge_shard_results([shard1, shard2])

        # Weighted mean: (mean1 * 4 + mean2 * 6) / 10
        expected_mean = (np.mean([0.9, 0.8, 0.3, 0.95]) * 4 + np.mean([0.4, 0.2, 0.85, 0.7, 0.6, 0.45]) * 6) / 10
        assert abs(merged["confidence_stats"]["mean"] - expected_mean) < 1e-6

        # Global min/max
        assert merged["confidence_stats"]["min"] == 0.2
        assert merged["confidence_stats"]["max"] == 0.95

        # Low confidence count is sum
        assert merged["confidence_stats"]["low_confidence_count"] == 4  # 1 + 3
        assert abs(merged["confidence_stats"]["low_confidence_percentage"] - 4 / 10 * 100) < 1e-4

    def test_merge_median_std_are_approximate(self):
        """Median and std come from shard_results[0] — they are not recomputed."""
        shard1 = make_shard_result({"en": 4}, [0.1, 0.2, 0.3, 0.4])
        shard2 = make_shard_result({"en": 4}, [0.8, 0.85, 0.9, 0.95])

        merged = merge_shard_results([shard1, shard2])

        # Median and std are taken from shard_results[0], not recomputed
        assert merged["confidence_stats"]["median"] == shard1["confidence_stats"]["median"]
        assert merged["confidence_stats"]["std"] == shard1["confidence_stats"]["std"]

    def test_merge_single_shard(self):
        """Merging a list of one shard should produce equivalent results."""
        shard = make_shard_result(
            {"en": 5, "de": 3},
            [0.9, 0.85, 0.7, 0.95, 0.88, 0.92, 0.6, 0.75],
            source_counts={"wiki": {"en": 5, "de": 3}},
            domain_counts={"general": {"en": 5, "de": 3}},
        )

        merged = merge_shard_results([shard])

        assert merged["total_samples"] == shard["total_samples"]
        assert merged["language_counts"] == shard["language_counts"]
        assert merged["language_percentages"] == shard["language_percentages"]
        assert merged["confidence_stats"]["mean"] == shard["confidence_stats"]["mean"]
        assert merged["confidence_stats"]["median"] == shard["confidence_stats"]["median"]
        assert merged["confidence_stats"]["std"] == shard["confidence_stats"]["std"]
        assert merged["confidence_stats"]["min"] == shard["confidence_stats"]["min"]
        assert merged["confidence_stats"]["max"] == shard["confidence_stats"]["max"]
        assert merged["per_source_dataset"] == shard["per_source_dataset"]
        assert merged["per_domain"] == shard["per_domain"]
        assert merged["num_shards_merged"] == 1

    def test_merge_preserves_language_order(self):
        """After merge, language_counts should be sorted by count descending."""
        shard1 = make_shard_result({"en": 1, "de": 5, "fr": 3}, [0.9] * 9)
        shard2 = make_shard_result({"en": 10, "de": 1, "fr": 2}, [0.9] * 13)

        merged = merge_shard_results([shard1, shard2])

        counts = list(merged["language_counts"].values())
        assert counts == sorted(counts, reverse=True), "language_counts should be sorted descending"


# --- Sharded processing tests ---


class TestShardedProcessing:
    @pytest.fixture()
    def small_dataset(self):
        """Create a small dataset with 40 samples for shard testing."""
        texts_and_langs = [
            ("This is an English text that should be long enough for detection.", "en"),
            (
                "Dies ist ein deutscher Text, der lang genug für die Erkennung sein sollte.",
                "de",
            ),
            (
                "Ceci est un texte français qui devrait être assez long pour la détection.",
                "fr",
            ),
            (
                "Este es un texto en español que debería ser lo suficientemente largo.",
                "es",
            ),
        ] * 10  # 40 samples

        records = []
        for i, (text, _) in enumerate(texts_and_langs):
            records.append(
                {
                    "messages": make_messages([text]),
                    "source_dataset": "test",
                    "domain": "general",
                    "id": str(i),
                }
            )
        return Dataset.from_list(records)

    def test_shards_cover_full_dataset(self, small_dataset):
        """Process 4 shards; total samples across shards equals 40 with no overlap."""
        num_shards = 4
        all_ids = []
        total_samples = 0

        for shard_idx in range(num_shards):
            shard_ds = small_dataset.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
            total_samples += len(shard_ds)
            all_ids.extend(shard_ds["id"])

        assert total_samples == 40
        # No duplicate IDs
        assert len(set(all_ids)) == 40

    def test_sharded_merge_matches_single(self, small_dataset):
        """Process same dataset with 1 shard vs 4 shards + merge; language_counts should match."""
        # Single-pass processing
        ds_single = small_dataset.map(extract_user_text, num_proc=1)
        ds_single = ds_single.map(detect_languages_batch, batched=True, batch_size=10, num_proc=1)
        result_single = aggregate_results(ds_single)

        # Sharded processing
        num_shards = 4
        shard_results = []
        for shard_idx in range(num_shards):
            shard_ds = small_dataset.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
            shard_ds = shard_ds.map(extract_user_text, num_proc=1)
            shard_ds = shard_ds.map(detect_languages_batch, batched=True, batch_size=10, num_proc=1)
            shard_results.append(aggregate_results(shard_ds))

        merged = merge_shard_results(shard_results)

        assert merged["total_samples"] == result_single["total_samples"]
        assert merged["language_counts"] == result_single["language_counts"]
        assert merged["language_percentages"] == result_single["language_percentages"]
        assert merged["per_source_dataset"] == result_single["per_source_dataset"]
        assert merged["per_domain"] == result_single["per_domain"]
        # Confidence: mean should match exactly (weighted mean of equal-sized shards)
        assert (
            abs(
                merged["confidence_stats"]["mean"]
                - result_single["confidence_stats"]["mean"]
            )
            < 1e-6
        )
        assert merged["confidence_stats"]["min"] == result_single["confidence_stats"]["min"]
        assert merged["confidence_stats"]["max"] == result_single["confidence_stats"]["max"]
        assert (
            merged["confidence_stats"]["low_confidence_count"]
            == result_single["confidence_stats"]["low_confidence_count"]
        )


# --- Edge case tests ---


class TestMergeEdgeCases:
    def test_merge_empty_list_raises(self):
        """Merging an empty list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            merge_shard_results([])

    def test_merge_disjoint_languages(self):
        """Shards with completely disjoint language sets merge correctly."""
        shard1 = make_shard_result({"ja": 5}, [0.9] * 5)
        shard2 = make_shard_result({"zh": 3}, [0.8] * 3)

        merged = merge_shard_results([shard1, shard2])

        assert merged["total_samples"] == 8
        assert merged["language_counts"] == {"ja": 5, "zh": 3}
        assert merged["language_percentages"]["ja"] == round(5 / 8 * 100, 4)

    def test_merge_all_unknown_languages(self):
        """Shards where all detections are 'unknown' merge correctly."""
        shard1 = make_shard_result({"unknown": 10}, [0.0] * 10)
        shard2 = make_shard_result({"unknown": 5}, [0.0] * 5)

        merged = merge_shard_results([shard1, shard2])

        assert merged["language_counts"] == {"unknown": 15}
        assert merged["language_percentages"]["unknown"] == 100.0

    def test_merge_low_confidence_languages(self):
        """Low-confidence language counts are merged across shards."""
        shard1 = make_shard_result({"en": 3, "de": 2}, [0.9, 0.3, 0.4, 0.8, 0.2])
        shard1["confidence_stats"]["low_confidence_languages"] = {"en": 1, "de": 1}
        shard2 = make_shard_result({"en": 2, "fr": 1}, [0.4, 0.9, 0.3])
        shard2["confidence_stats"]["low_confidence_languages"] = {"en": 1, "fr": 1}

        merged = merge_shard_results([shard1, shard2])

        assert merged["confidence_stats"]["low_confidence_languages"]["en"] == 2
        assert merged["confidence_stats"]["low_confidence_languages"]["de"] == 1
        assert merged["confidence_stats"]["low_confidence_languages"]["fr"] == 1

    def test_merge_missing_per_source_key(self):
        """Shards that lack per_source_dataset entirely merge cleanly."""
        shard1 = make_shard_result({"en": 5}, [0.9] * 5)
        del shard1["per_source_dataset"]
        shard2 = make_shard_result(
            {"en": 3}, [0.8] * 3, source_counts={"wiki": {"en": 3}}
        )

        merged = merge_shard_results([shard1, shard2])
        assert merged["per_source_dataset"]["wiki"]["en"] == 3

    def test_merge_num_shards_merged_field(self):
        """Merged result includes num_shards_merged with correct count."""
        shards = [make_shard_result({"en": 2}, [0.9] * 2) for _ in range(7)]
        merged = merge_shard_results(shards)
        assert merged["num_shards_merged"] == 7


# --- cmd_merge integration tests ---


class TestCmdMerge:
    def test_merge_from_disk(self, tmp_path):
        """cmd_merge loads shard JSONs, merges, and writes the result."""
        shard1 = make_shard_result({"en": 5}, [0.9] * 5)
        shard2 = make_shard_result({"de": 3}, [0.8] * 3)

        (tmp_path / "dolci_instruct_lang_dist_shard0.json").write_text(
            json.dumps(shard1)
        )
        (tmp_path / "dolci_instruct_lang_dist_shard1.json").write_text(
            json.dumps(shard2)
        )

        args = argparse.Namespace(output_dir=tmp_path, dataset="instruct")
        cmd_merge(args)

        merged_file = tmp_path / "dolci_instruct_lang_dist.json"
        assert merged_file.exists()
        merged = json.loads(merged_file.read_text())
        assert merged["total_samples"] == 8
        assert merged["language_counts"]["en"] == 5
        assert merged["language_counts"]["de"] == 3
        assert merged["dataset_key"] == "instruct"

    def test_merge_fallback_to_single_node(self, tmp_path):
        """When no shard files exist but a single-node file does, load it."""
        single = make_shard_result({"en": 10}, [0.9] * 10)
        single["dataset"] = "allenai/Dolci-Instruct-SFT"
        single["dataset_key"] = "instruct"
        (tmp_path / "dolci_instruct_lang_dist.json").write_text(json.dumps(single))

        args = argparse.Namespace(output_dir=tmp_path, dataset="instruct")
        cmd_merge(args)  # Should not crash

    def test_merge_no_files_does_not_crash(self, tmp_path):
        """When no shard or single files exist, warn and skip gracefully."""
        args = argparse.Namespace(output_dir=tmp_path, dataset="instruct")
        cmd_merge(args)  # Should not crash
        # No merged output file should be created
        assert not (tmp_path / "dolci_instruct_lang_dist.json").exists()
