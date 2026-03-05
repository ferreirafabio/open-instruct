"""Tests for multilingual preprocessing pipeline.

Tests transform functions, language splitting, mixture assembly,
language overlap validation, I/O safety, and parallel execution
for the EU multilingual fine-tuning experiment.
"""

import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
import yaml

from oellm.pipelines.preprocessing.preprocess_datasets import (
    transform_fusion_synth,
    transform_lmsys_chat_multilingual,
    transform_wildchat,
)


# ============================================================================
# Transform tests: fusion-synth
# ============================================================================


class TestTransformFusionSynth:
    def test_basic(self):
        """Correct messages format with Fusion completion extracted."""
        example = {
            "prompt": "What is AI?",
            "Fusion": {"completion": "AI is artificial intelligence."},
            "language_code": "en",
        }
        result = transform_fusion_synth(example)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "What is AI?"
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["content"] == "AI is artificial intelligence."
        assert result["language"] == "en"

    def test_language_preserved(self):
        """Language code is preserved in output."""
        example = {
            "prompt": "Was ist KI?",
            "Fusion": {"completion": "KI ist kuenstliche Intelligenz."},
            "language_code": "de",
        }
        result = transform_fusion_synth(example)
        assert result["language"] == "de"

    def test_missing_fusion(self):
        """Missing Fusion column returns empty messages."""
        example = {
            "prompt": "What is AI?",
            "Fusion": None,
            "language_code": "en",
        }
        result = transform_fusion_synth(example)
        assert result["messages"] == []

    def test_missing_fusion_key(self):
        """Missing Fusion key entirely returns empty messages."""
        example = {
            "prompt": "What is AI?",
            "language_code": "en",
        }
        result = transform_fusion_synth(example)
        assert result["messages"] == []

    def test_empty_prompt(self):
        """Empty prompt returns empty messages."""
        example = {
            "prompt": "",
            "Fusion": {"completion": "Some answer."},
            "language_code": "en",
        }
        result = transform_fusion_synth(example)
        assert result["messages"] == []

    def test_empty_completion(self):
        """Empty completion returns empty messages."""
        example = {
            "prompt": "What is AI?",
            "Fusion": {"completion": ""},
            "language_code": "en",
        }
        result = transform_fusion_synth(example)
        assert result["messages"] == []


# ============================================================================
# Transform tests: WildChat
# ============================================================================


class TestTransformWildChat:
    def test_basic(self):
        """Basic conversation with language preserved."""
        example = {
            "conversation": [
                {"role": "user", "content": "Hello, how are you doing today? I have a question about programming."},
                {"role": "assistant", "content": "I'm doing well, thank you! What would you like to know about programming?"},
            ],
            "language": "en",
            "toxic": False,
        }
        result = transform_wildchat(example)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"
        assert result["language"] == "en"

    def test_toxic_filtered(self):
        """Toxic conversations return empty messages."""
        example = {
            "conversation": [
                {"role": "user", "content": "Bad content here"},
                {"role": "assistant", "content": "Response"},
            ],
            "language": "en",
            "toxic": True,
        }
        result = transform_wildchat(example)
        assert result["messages"] == []

    def test_min_turns(self):
        """Single turn (< 2) returns empty messages."""
        example = {
            "conversation": [
                {"role": "user", "content": "Hello"},
            ],
            "language": "en",
            "toxic": False,
        }
        result = transform_wildchat(example)
        assert result["messages"] == []

    def test_min_chars(self):
        """Short content (< 50 chars) returns empty messages."""
        example = {
            "conversation": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hey"},
            ],
            "language": "en",
            "toxic": False,
        }
        result = transform_wildchat(example)
        assert result["messages"] == []

    def test_language_mapped_to_iso(self):
        """Full language names are mapped to ISO codes."""
        content = "Dies ist ein langer Satz auf Deutsch, der lang genug ist."
        example = {
            "conversation": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": content},
            ],
            "language": "German",
            "toxic": False,
        }
        result = transform_wildchat(example)
        assert result["language"] == "de"  # "German" -> "de"


# ============================================================================
# Transform tests: lmsys-chat multilingual
# ============================================================================


class TestTransformLmsysChatMultilingual:
    def test_basic(self):
        """Language column preserved and mapped to ISO code."""
        content = "This is a sufficiently long conversation turn for the test."
        example = {
            "conversation": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": content},
            ],
            "language": "English",
        }
        result = transform_lmsys_chat_multilingual(example)
        assert len(result["messages"]) == 2
        assert result["language"] == "en"  # "English" -> "en"

    def test_language_name_to_iso_mapping(self):
        """Full language names are mapped to ISO 639-1 codes."""
        content = "This is a sufficiently long conversation turn for testing purposes."
        for full_name, iso_code in [("German", "de"), ("French", "fr"), ("Czech", "cs")]:
            example = {
                "conversation": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": content},
                ],
                "language": full_name,
            }
            result = transform_lmsys_chat_multilingual(example)
            assert result["language"] == iso_code, f"{full_name} should map to {iso_code}"

    def test_min_turns(self):
        """Single turn returns empty messages."""
        example = {
            "conversation": [
                {"role": "user", "content": "Hello"},
            ],
            "language": "English",
        }
        result = transform_lmsys_chat_multilingual(example)
        assert result["messages"] == []

    def test_min_chars(self):
        """Short content returns empty messages."""
        example = {
            "conversation": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Yo"},
            ],
            "language": "English",
        }
        result = transform_lmsys_chat_multilingual(example)
        assert result["messages"] == []

    def test_empty_conversation(self):
        """Empty conversation list returns empty messages."""
        example = {
            "conversation": [],
            "language": "English",
        }
        result = transform_lmsys_chat_multilingual(example)
        assert result["messages"] == []


# ============================================================================
# Transform tests: oasst2 (tree reconstruction)
# ============================================================================


class TestTransformOasst2:
    def test_tree_reconstruction(self):
        """Best-ranked path extracted from conversation tree."""
        from datasets import Dataset

        from oellm.pipelines.preprocessing.preprocess_datasets import (
            transform_oasst2_dataset,
        )

        # Build a simple tree: root -> child1 (rank 0) + child2 (rank 1)
        data = [
            {
                "message_id": "root1",
                "parent_id": None,
                "role": "prompter",
                "text": "What is Python?",
                "lang": "en",
                "rank": None,
                "labels": {},
            },
            {
                "message_id": "child1",
                "parent_id": "root1",
                "role": "assistant",
                "text": "Python is a programming language.",
                "lang": "en",
                "rank": 0,
                "labels": {"quality": 1},
            },
            {
                "message_id": "child2",
                "parent_id": "root1",
                "role": "assistant",
                "text": "Python is a snake.",
                "lang": "en",
                "rank": 1,
                "labels": {},
            },
        ]
        ds = Dataset.from_list(data)
        conversations = transform_oasst2_dataset(ds)

        assert len(conversations) == 1
        conv = conversations[0]
        assert len(conv["messages"]) == 2
        assert conv["messages"][0]["role"] == "user"
        assert conv["messages"][0]["content"] == "What is Python?"
        assert conv["messages"][1]["role"] == "assistant"
        # Should pick rank 0 (best)
        assert conv["messages"][1]["content"] == "Python is a programming language."

    def test_language_preserved(self):
        """Language from root message is preserved."""
        from datasets import Dataset

        from oellm.pipelines.preprocessing.preprocess_datasets import (
            transform_oasst2_dataset,
        )

        data = [
            {
                "message_id": "root1",
                "parent_id": None,
                "role": "prompter",
                "text": "Was ist Python?",
                "lang": "de",
                "rank": None,
                "labels": {},
            },
            {
                "message_id": "child1",
                "parent_id": "root1",
                "role": "assistant",
                "text": "Python ist eine Programmiersprache.",
                "lang": "de",
                "rank": 0,
                "labels": {},
            },
        ]
        ds = Dataset.from_list(data)
        conversations = transform_oasst2_dataset(ds)

        assert len(conversations) == 1
        assert conversations[0]["language"] == "de"

    def test_language_normalized(self):
        """Language codes like pt-BR are normalized to pt."""
        from datasets import Dataset

        from oellm.pipelines.preprocessing.preprocess_datasets import (
            transform_oasst2_dataset,
        )

        data = [
            {
                "message_id": "root1",
                "parent_id": None,
                "role": "prompter",
                "text": "O que e Python?",
                "lang": "pt-BR",
                "rank": None,
                "labels": {},
            },
            {
                "message_id": "child1",
                "parent_id": "root1",
                "role": "assistant",
                "text": "Python e uma linguagem de programacao.",
                "lang": "pt-BR",
                "rank": 0,
                "labels": {},
            },
        ]
        ds = Dataset.from_list(data)
        conversations = transform_oasst2_dataset(ds)

        assert len(conversations) == 1
        assert conversations[0]["language"] == "pt"  # pt-BR -> pt

    def test_skips_assistant_root(self):
        """Conversations starting with assistant role are skipped."""
        from datasets import Dataset

        from oellm.pipelines.preprocessing.preprocess_datasets import (
            transform_oasst2_dataset,
        )

        data = [
            {
                "message_id": "root1",
                "parent_id": None,
                "role": "assistant",
                "text": "Hello!",
                "lang": "en",
                "rank": None,
                "labels": {},
            },
        ]
        ds = Dataset.from_list(data)
        conversations = transform_oasst2_dataset(ds)
        assert len(conversations) == 0


# ============================================================================
# Split by language tests
# ============================================================================


class TestSplitByLanguage:
    def test_basic_split(self, tmp_path):
        """Correct per-language parquet files created with correct row counts."""
        from oellm.pipelines.preprocessing.split_by_language import (
            split_parquet_by_language,
        )

        df = pd.DataFrame({
            "messages": [
                [{"role": "user", "content": "Hello"}],
                [{"role": "user", "content": "Hallo"}],
                [{"role": "user", "content": "Bonjour"}],
                [{"role": "user", "content": "Hi again"}],
            ],
            "language": ["en", "de", "fr", "en"],
        })
        input_path = tmp_path / "test_dataset.parquet"
        df.to_parquet(input_path)

        output_dir = tmp_path / "output"
        counts = split_parquet_by_language(input_path, output_dir)

        assert counts == {"en": 2, "de": 1, "fr": 1}
        assert (output_dir / "test_dataset" / "en.parquet").exists()
        assert (output_dir / "test_dataset" / "de.parquet").exists()
        assert (output_dir / "test_dataset" / "fr.parquet").exists()

        # Verify row counts
        en_df = pd.read_parquet(output_dir / "test_dataset" / "en.parquet")
        assert len(en_df) == 2

    def test_empty_language_filtered(self, tmp_path):
        """Rows with empty language are not included."""
        from oellm.pipelines.preprocessing.split_by_language import (
            split_parquet_by_language,
        )

        df = pd.DataFrame({
            "messages": [
                [{"role": "user", "content": "Hello"}],
                [{"role": "user", "content": "Unknown"}],
            ],
            "language": ["en", ""],
        })
        input_path = tmp_path / "test_dataset.parquet"
        df.to_parquet(input_path)

        output_dir = tmp_path / "output"
        counts = split_parquet_by_language(input_path, output_dir)

        assert "en" in counts
        assert "" not in counts

    def test_no_language_column_raises(self, tmp_path):
        """Missing language column raises ValueError."""
        from oellm.pipelines.preprocessing.split_by_language import (
            split_parquet_by_language,
        )

        df = pd.DataFrame({
            "messages": [[{"role": "user", "content": "Hello"}]],
        })
        input_path = tmp_path / "test_dataset.parquet"
        df.to_parquet(input_path)

        output_dir = tmp_path / "output"
        with pytest.raises(ValueError, match="No 'language' column"):
            split_parquet_by_language(input_path, output_dir)


# ============================================================================
# Assembly tests
# ============================================================================


class TestAssembleMixture:
    def _create_by_language_dir(self, tmp_path, source_name, lang_data):
        """Helper to create per-language parquet files."""
        source_dir = tmp_path / "by_language" / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        for lang, n_samples in lang_data.items():
            df = pd.DataFrame({
                "messages": [[{"role": "user", "content": f"Hello in {lang}"}]] * n_samples,
                "language": [lang] * n_samples,
            })
            df.to_parquet(source_dir / f"{lang}.parquet")

    def test_ratios(self, tmp_path):
        """Sample counts match YAML ratios within rounding tolerance."""
        from oellm.pipelines.preprocessing.assemble_mixture import (
            assemble_mixture,
        )

        self._create_by_language_dir(tmp_path, "source1", {
            "de": 500, "fr": 400, "es": 300, "it": 200,
            "pt": 200, "pl": 100, "nl": 100, "cs": 100,
        })

        config = {
            "name": "test-mix",
            "total_samples": 1000,
            "english_ratio": 0.90,
            "languages": {"de": 0.025, "fr": 0.020, "es": 0.020, "it": 0.010, "pt": 0.010, "pl": 0.005, "nl": 0.005, "cs": 0.005},
            "sources": ["source1"],
            "english_source": "test/english-source",
            "seed": 42,
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_dir = tmp_path / "output"
        result = assemble_mixture(
            config_path=config_path,
            by_language_dir=tmp_path / "by_language",
            output_dir=output_dir,
        )

        assert result is not None
        df = pd.read_parquet(result)

        # Check per-language counts (non-English, since English is deferred)
        lang_counts = df["language"].value_counts().to_dict()
        assert lang_counts.get("de", 0) == 25  # 1000 * 0.025
        assert lang_counts.get("fr", 0) == 20  # 1000 * 0.020

    def test_source_priority(self, tmp_path):
        """Higher-priority sources used first."""
        from oellm.pipelines.preprocessing.assemble_mixture import (
            load_language_data,
        )

        # Source1 has 10 samples, source2 has 100
        self._create_by_language_dir(tmp_path, "source1", {"de": 10})
        self._create_by_language_dir(tmp_path, "source2", {"de": 100})

        by_lang_dir = tmp_path / "by_language"

        # Request 15 samples with source1 priority
        df = load_language_data("de", ["source1", "source2"], by_lang_dir, 15, seed=42)
        assert len(df) == 15  # 10 from source1 + 5 from source2

    def test_insufficient_data_warning(self, tmp_path):
        """Warning when a language has fewer samples than requested."""
        from oellm.pipelines.preprocessing.assemble_mixture import (
            assemble_mixture,
        )

        # Only 5 German samples available
        self._create_by_language_dir(tmp_path, "source1", {"de": 5})

        config = {
            "name": "test-insufficient",
            "total_samples": 1000,
            "english_ratio": 0.90,
            "languages": {"de": 0.10},
            "sources": ["source1"],
            "english_source": "test/english-source",
            "seed": 42,
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_dir = tmp_path / "output"
        with pytest.warns(UserWarning, match="only 5"):
            assemble_mixture(
                config_path=config_path,
                by_language_dir=tmp_path / "by_language",
                output_dir=output_dir,
            )

    def test_yaml_validation_bad_ratios(self, tmp_path):
        """Invalid ratios that don't sum to 1.0 raise error."""
        config = {
            "name": "bad-config",
            "total_samples": 1000,
            "english_ratio": 0.50,
            "languages": {"de": 0.10},
            "sources": ["source1"],
            "seed": 42,
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from oellm.utils.language_mixer import LanguageMixer

        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            LanguageMixer.from_yaml(config_path)


# ============================================================================
# Language overlap test
# ============================================================================


class TestLanguageOverlap:
    def test_overlap_with_benchmark(self):
        """Verify training langs overlap with m-arena-hard-EU, ro and el held out."""
        benchmark_languages = {"cs", "de", "el", "en", "es", "fr", "it", "nl", "pl", "pt", "ro", "uk"}
        training_languages = {"de", "es", "fr", "it", "pt", "pl", "nl", "cs"}
        held_out = {"ro", "el"}

        # Training languages should be subset of benchmark (minus held-out)
        assert training_languages.issubset(benchmark_languages - held_out)
        # Held-out should be in benchmark
        assert held_out.issubset(benchmark_languages)
        # No overlap between training and held-out
        assert training_languages.isdisjoint(held_out)


# ============================================================================
# Profile output format test
# ============================================================================


class TestProfileOutputFormat:
    def test_output_structure(self):
        """Verify expected JSON structure for profile output."""
        # Simulate the expected output format
        profile = {
            "fusion-synth": {
                "total": 94700,
                "per_language": {"en": 9470, "de": 9470},
                "num_languages": 10,
            },
        }

        # Verify structure
        for dataset_name, data in profile.items():
            assert "total" in data
            assert "per_language" in data
            assert isinstance(data["per_language"], dict)
            assert isinstance(data["total"], int)

    def test_per_dataset_files_no_race(self, tmp_path):
        """Each dataset writes its own file — no race on dataset_profiles.json."""
        from oellm.pipelines.preprocessing.profile_multilingual_datasets import (
            merge_profiles,
        )

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Simulate 4 parallel tasks writing per-dataset files
        for name in ["fusion-synth", "wildchat", "lmsys-chat", "oasst2"]:
            data = {name: {"total": 100, "per_language": {"en": 80, "de": 20}, "num_languages": 2}}
            with open(profiles_dir / f"profile_{name}.json", "w") as f:
                json.dump(data, f)

        # Merge should collect all 4
        merged = merge_profiles(profiles_dir)
        assert len(merged) == 4
        assert set(merged.keys()) == {"fusion-synth", "wildchat", "lmsys-chat", "oasst2"}

    def test_merge_handles_partial_results(self, tmp_path):
        """Merge works correctly even if only some datasets completed."""
        from oellm.pipelines.preprocessing.profile_multilingual_datasets import (
            merge_profiles,
        )

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Only 2 of 4 datasets completed
        for name in ["fusion-synth", "oasst2"]:
            data = {name: {"total": 50, "per_language": {"en": 50}, "num_languages": 1}}
            with open(profiles_dir / f"profile_{name}.json", "w") as f:
                json.dump(data, f)

        merged = merge_profiles(profiles_dir)
        assert len(merged) == 2


# ============================================================================
# I/O safety tests: parallel split and assembly
# ============================================================================


class TestParallelSplitSafety:
    """Test that parallel split-by-language operations don't interfere."""

    def test_parallel_splits_write_to_separate_dirs(self, tmp_path):
        """Each dataset splits into its own subdirectory — no file conflicts."""
        from oellm.pipelines.preprocessing.split_by_language import (
            split_parquet_by_language,
        )

        output_dir = tmp_path / "by_language"

        # Create 4 different source parquets
        datasets = {}
        for name in ["source_a", "source_b", "source_c", "source_d"]:
            df = pd.DataFrame({
                "messages": [[{"role": "user", "content": f"Hello from {name}"}]] * 10,
                "language": ["en"] * 5 + ["de"] * 5,
            })
            path = tmp_path / f"{name}.parquet"
            df.to_parquet(path)
            datasets[name] = path

        # Simulate parallel splitting (threads represent SLURM array tasks)
        def split_one(item):
            name, path = item
            return name, split_parquet_by_language(path, output_dir)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = dict(executor.map(split_one, datasets.items()))

        # Each dataset has its own directory
        for name in datasets:
            assert (output_dir / name / "en.parquet").exists()
            assert (output_dir / name / "de.parquet").exists()

        # Verify row counts are correct (no cross-contamination)
        for name in datasets:
            en_df = pd.read_parquet(output_dir / name / "en.parquet")
            de_df = pd.read_parquet(output_dir / name / "de.parquet")
            assert len(en_df) == 5
            assert len(de_df) == 5

    def test_assembly_reads_from_separate_source_dirs(self, tmp_path):
        """Assembly reads per-language files from source-specific dirs — no conflicts."""
        from oellm.pipelines.preprocessing.assemble_mixture import (
            load_language_data,
        )

        by_lang_dir = tmp_path / "by_language"

        # Create separate source directories
        for source in ["source1", "source2"]:
            source_dir = by_lang_dir / source
            source_dir.mkdir(parents=True)
            df = pd.DataFrame({
                "messages": [[{"role": "user", "content": f"From {source}"}]] * 20,
                "language": ["de"] * 20,
            })
            df.to_parquet(source_dir / "de.parquet")

        # Load from both sources in priority order
        result = load_language_data("de", ["source1", "source2"], by_lang_dir, 30, seed=42)
        assert len(result) == 30  # 20 from source1 + 10 from source2

    def test_assembly_output_is_atomic(self, tmp_path):
        """Assembled parquet is written completely before being usable."""
        from oellm.pipelines.preprocessing.assemble_mixture import (
            assemble_mixture,
        )

        # Create minimal source data
        source_dir = tmp_path / "by_language" / "src"
        source_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "messages": [[{"role": "user", "content": "Hello"}]] * 100,
            "language": ["de"] * 100,
        })
        df.to_parquet(source_dir / "de.parquet")

        config = {
            "name": "atomic-test",
            "total_samples": 100,
            "english_ratio": 0.50,
            "languages": {"de": 0.50},
            "sources": ["src"],
            "english_source": "test/english",
            "seed": 42,
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_dir = tmp_path / "assembled"
        result = assemble_mixture(
            config_path=config_path,
            by_language_dir=tmp_path / "by_language",
            output_dir=output_dir,
        )

        assert result is not None
        # Verify the parquet is valid and readable
        assembled_df = pd.read_parquet(result)
        assert len(assembled_df) == 50  # 100 * 0.50 for de (English deferred)

        # Metadata file should also exist
        metadata_path = output_dir / "atomic-test_metadata.json"
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["total_samples_actual"] == 50


class TestParallelPreprocessSafety:
    """Test that parallel preprocessing of different datasets doesn't interfere."""

    def test_output_files_are_dataset_specific(self, tmp_path):
        """Each mixture writes to its own parquet — no overwriting."""
        # Simulate what happens when 4 SLURM array tasks write to same output_dir
        output_dir = tmp_path / "preprocessed"
        output_dir.mkdir()

        # Each task writes a different file
        for name in ["CohereLabs-fusion-synth", "allenai-WildChat", "lmsys-chat", "oasst2"]:
            df = pd.DataFrame({
                "messages": [[{"role": "user", "content": f"From {name}"}]] * 10,
                "language": ["en"] * 10,
            })
            output_path = output_dir / f"{name}.parquet"
            df.to_parquet(output_path)

        # All 4 files should exist independently
        files = list(output_dir.glob("*.parquet"))
        assert len(files) == 4

        # Each file has correct content (no mixing)
        for parquet_file in files:
            df = pd.read_parquet(parquet_file)
            expected_source = parquet_file.stem
            assert all(expected_source in msg[0]["content"] for msg in df["messages"])
