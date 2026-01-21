"""Unit and property tests for the language mixer.

These tests verify:
1. Ratio validation (must sum to 1.0)
2. Explicit language ratios are respected
3. "Others" ratio is distributed evenly
4. YAML config parsing works correctly
5. Property-based tests for edge cases
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st, assume

from oellm.utils.language_mixer import LanguageMixer, LanguageMixConfig


# All 23 EU languages (excluding English)
EU_LANGUAGES = [
    "bg", "cs", "da", "de", "el", "es", "et", "fi",
    "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt",
    "nl", "pl", "pt", "ro", "sk", "sl", "sv"
]


class TestLanguageMixConfigValidation:
    """Tests for LanguageMixConfig validation."""

    def test_valid_config_100_english(self):
        """100% English is valid."""
        config = LanguageMixConfig(
            english_ratio=1.0,
            explicit_languages={},
            others_ratio=0.0,
            datasets=["dataset1.parquet"]
        )
        mixer = LanguageMixer(config)
        assert mixer is not None

    def test_valid_config_90_10_split(self):
        """90% English, 10% others is valid."""
        config = LanguageMixConfig(
            english_ratio=0.9,
            explicit_languages={},
            others_ratio=0.1,
            datasets=["dataset1.parquet"]
        )
        mixer = LanguageMixer(config)
        assert mixer is not None

    def test_valid_config_with_explicit_languages(self):
        """Config with explicit languages summing correctly is valid."""
        config = LanguageMixConfig(
            english_ratio=0.8,
            explicit_languages={"de": 0.05, "fr": 0.03, "es": 0.02},
            others_ratio=0.1,
            datasets=["dataset1.parquet"]
        )
        mixer = LanguageMixer(config)
        assert mixer is not None

    def test_invalid_config_ratios_exceed_one(self):
        """Ratios exceeding 1.0 raise ValueError."""
        config = LanguageMixConfig(
            english_ratio=0.9,
            explicit_languages={"de": 0.1},
            others_ratio=0.1,
            datasets=["dataset1.parquet"]
        )
        with pytest.raises(ValueError, match="must sum to 1.0"):
            LanguageMixer(config)

    def test_invalid_config_ratios_below_one(self):
        """Ratios below 1.0 raise ValueError."""
        config = LanguageMixConfig(
            english_ratio=0.5,
            explicit_languages={},
            others_ratio=0.2,
            datasets=["dataset1.parquet"]
        )
        with pytest.raises(ValueError, match="must sum to 1.0"):
            LanguageMixer(config)

    def test_invalid_config_negative_ratio(self):
        """Negative ratios raise ValueError."""
        config = LanguageMixConfig(
            english_ratio=1.1,
            explicit_languages={"de": -0.1},
            others_ratio=0.0,
            datasets=["dataset1.parquet"]
        )
        with pytest.raises(ValueError):
            LanguageMixer(config)

    def test_invalid_explicit_language_not_in_eu(self):
        """Explicit language not in EU list raises ValueError."""
        config = LanguageMixConfig(
            english_ratio=0.9,
            explicit_languages={"xx": 0.1},  # Invalid language code
            others_ratio=0.0,
            datasets=["dataset1.parquet"]
        )
        with pytest.raises(ValueError, match="not a valid EU language"):
            LanguageMixer(config)

    def test_english_cannot_be_in_explicit_languages(self):
        """English in explicit_languages raises ValueError."""
        config = LanguageMixConfig(
            english_ratio=0.8,
            explicit_languages={"en": 0.1, "de": 0.1},
            others_ratio=0.0,
            datasets=["dataset1.parquet"]
        )
        with pytest.raises(ValueError, match="English.*english_ratio"):
            LanguageMixer(config)


class TestComputeSamplesPerLanguage:
    """Tests for computing sample counts per language."""

    def test_100_english(self):
        """100% English returns only English samples."""
        config = LanguageMixConfig(
            english_ratio=1.0,
            explicit_languages={},
            others_ratio=0.0,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        result = mixer.compute_samples_per_language(10000)

        assert result["en"] == 10000
        # No other languages should have samples
        for lang in EU_LANGUAGES:
            assert result.get(lang, 0) == 0

    def test_90_10_split_with_others(self):
        """90% English, 10% others distributed evenly."""
        config = LanguageMixConfig(
            english_ratio=0.9,
            explicit_languages={},
            others_ratio=0.1,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        result = mixer.compute_samples_per_language(23000)  # Divisible by 23

        assert result["en"] == 20700  # 90% of 23000
        # Each of 23 languages gets 1/23 of 10% = ~100 each
        total_eu = sum(result[lang] for lang in EU_LANGUAGES)
        assert total_eu == 2300  # 10% of 23000

    def test_explicit_languages_respected(self):
        """Explicit language ratios produce correct sample counts."""
        config = LanguageMixConfig(
            english_ratio=0.8,
            explicit_languages={"de": 0.1, "fr": 0.05},
            others_ratio=0.05,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        result = mixer.compute_samples_per_language(10000)

        assert result["en"] == 8000
        assert result["de"] == 1000
        assert result["fr"] == 500
        # Others share remaining 5%
        others = [l for l in EU_LANGUAGES if l not in ["de", "fr"]]
        per_other = int(10000 * 0.05 / len(others))
        for lang in others:
            assert result[lang] == per_other

    def test_others_divided_evenly(self):
        """Others ratio is divided evenly among non-explicit EU languages."""
        config = LanguageMixConfig(
            english_ratio=0.79,
            explicit_languages={"de": 0.1, "fr": 0.05, "es": 0.03},
            others_ratio=0.03,  # 3% for remaining 20 languages
            datasets=[]
        )
        mixer = LanguageMixer(config)

        result = mixer.compute_samples_per_language(20000)

        # Remaining languages
        others = [l for l in EU_LANGUAGES if l not in ["de", "fr", "es"]]
        assert len(others) == 20

        # Each should get 3% / 20 = 0.15% of 20000 = 30
        for lang in others:
            assert result[lang] == 30

    def test_small_total_samples_rounding(self):
        """Small total samples handles rounding correctly."""
        config = LanguageMixConfig(
            english_ratio=0.9,
            explicit_languages={},
            others_ratio=0.1,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        result = mixer.compute_samples_per_language(100)

        # Total should not exceed input
        total = sum(result.values())
        assert total <= 100

    def test_zero_others_ratio(self):
        """Zero others ratio gives 0 samples to non-explicit languages."""
        config = LanguageMixConfig(
            english_ratio=0.9,
            explicit_languages={"de": 0.1},
            others_ratio=0.0,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        result = mixer.compute_samples_per_language(10000)

        assert result["en"] == 9000
        assert result["de"] == 1000
        # All others should be 0
        for lang in EU_LANGUAGES:
            if lang != "de":
                assert result.get(lang, 0) == 0


class TestYAMLConfigParsing:
    """Tests for loading config from YAML files."""

    def test_parse_simple_config(self, tmp_path: Path):
        """Simple YAML config is parsed correctly."""
        config_content = """
english_ratio: 0.9
languages:
  others: 0.1
datasets:
  - dataset1.parquet
  - dataset2.parquet
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        mixer = LanguageMixer.from_yaml(config_file)

        assert mixer.config.english_ratio == 0.9
        assert mixer.config.others_ratio == 0.1
        assert len(mixer.config.datasets) == 2

    def test_parse_config_with_explicit_languages(self, tmp_path: Path):
        """YAML config with explicit languages is parsed correctly."""
        config_content = """
english_ratio: 0.8
languages:
  de: 0.05
  fr: 0.03
  es: 0.02
  others: 0.1
datasets:
  - dataset1.parquet
seed: 42
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        mixer = LanguageMixer.from_yaml(config_file)

        assert mixer.config.english_ratio == 0.8
        assert mixer.config.explicit_languages["de"] == 0.05
        assert mixer.config.explicit_languages["fr"] == 0.03
        assert mixer.config.explicit_languages["es"] == 0.02
        assert mixer.config.others_ratio == 0.1
        assert mixer.config.seed == 42

    def test_parse_100_english_config(self, tmp_path: Path):
        """100% English config is parsed correctly."""
        config_content = """
english_ratio: 1.0
languages: {}
datasets:
  - dataset1.parquet
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        mixer = LanguageMixer.from_yaml(config_file)

        assert mixer.config.english_ratio == 1.0
        assert mixer.config.others_ratio == 0.0
        assert len(mixer.config.explicit_languages) == 0

    def test_invalid_yaml_raises_error(self, tmp_path: Path):
        """Invalid YAML raises appropriate error."""
        config_content = """
english_ratio: 0.9
languages:
  - invalid yaml structure
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises((ValueError, TypeError)):
            LanguageMixer.from_yaml(config_file)

    def test_missing_required_field(self, tmp_path: Path):
        """Missing required field raises error."""
        config_content = """
languages:
  others: 0.1
datasets:
  - dataset1.parquet
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises((ValueError, KeyError)):
            LanguageMixer.from_yaml(config_file)


class TestLanguageMixerProperties:
    """Property-based tests using Hypothesis."""

    @given(
        en_ratio=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
        total=st.integers(min_value=1000, max_value=1000000)
    )
    @settings(max_examples=100)
    def test_total_samples_matches_input(self, en_ratio: float, total: int):
        """Sum of per-language samples is close to total (within rounding)."""
        others_ratio = 1.0 - en_ratio

        config = LanguageMixConfig(
            english_ratio=en_ratio,
            explicit_languages={},
            others_ratio=others_ratio,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        result = mixer.compute_samples_per_language(total)
        actual_total = sum(result.values())

        # Allow for small rounding differences (max 1 per language + English)
        max_rounding_error = len(EU_LANGUAGES) + 1
        assert abs(actual_total - total) <= max_rounding_error

    @given(
        explicit_ratios=st.dictionaries(
            keys=st.sampled_from(EU_LANGUAGES[:10]),  # Limit to first 10 for speed
            values=st.floats(min_value=0.001, max_value=0.05, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=5
        ),
        total=st.integers(min_value=10000, max_value=100000)
    )
    @settings(max_examples=50)
    def test_explicit_ratios_never_negative(self, explicit_ratios: dict, total: int):
        """No language ever gets negative samples."""
        # Calculate remaining ratio for English and others
        explicit_sum = sum(explicit_ratios.values())
        assume(explicit_sum < 0.9)  # Leave room for English

        en_ratio = 0.9 - explicit_sum
        others_ratio = 0.1

        # Ensure we have valid ratios
        assume(en_ratio >= 0.1)
        assume(abs(en_ratio + explicit_sum + others_ratio - 1.0) < 1e-6)

        config = LanguageMixConfig(
            english_ratio=en_ratio,
            explicit_languages=explicit_ratios,
            others_ratio=others_ratio,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        result = mixer.compute_samples_per_language(total)

        # No language should have negative samples
        for lang, count in result.items():
            assert count >= 0, f"Language {lang} has negative count: {count}"

    @given(total=st.integers(min_value=1, max_value=100))
    @settings(max_examples=50)
    def test_small_totals_dont_crash(self, total: int):
        """Very small total samples don't cause crashes or errors."""
        config = LanguageMixConfig(
            english_ratio=0.9,
            explicit_languages={},
            others_ratio=0.1,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        # Should not raise
        result = mixer.compute_samples_per_language(total)

        # Results should be valid
        assert isinstance(result, dict)
        assert all(isinstance(v, int) for v in result.values())
        assert all(v >= 0 for v in result.values())

    @given(
        en_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        others_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_invalid_ratios_are_rejected(self, en_ratio: float, others_ratio: float):
        """Invalid ratio combinations are always rejected."""
        assume(abs(en_ratio + others_ratio - 1.0) > 1e-6)  # Invalid ratios

        config = LanguageMixConfig(
            english_ratio=en_ratio,
            explicit_languages={},
            others_ratio=others_ratio,
            datasets=[]
        )

        with pytest.raises(ValueError):
            LanguageMixer(config)


class TestGetLanguagesForMixing:
    """Tests for getting the list of languages based on config."""

    def test_100_english_returns_only_english(self):
        """100% English config returns only English."""
        config = LanguageMixConfig(
            english_ratio=1.0,
            explicit_languages={},
            others_ratio=0.0,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        languages = mixer.get_languages_for_mixing()

        assert languages == ["en"]

    def test_with_others_returns_all_languages(self):
        """Config with others ratio returns all languages."""
        config = LanguageMixConfig(
            english_ratio=0.9,
            explicit_languages={},
            others_ratio=0.1,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        languages = mixer.get_languages_for_mixing()

        assert "en" in languages
        assert len(languages) == 24  # English + 23 EU languages

    def test_with_explicit_only(self):
        """Config with explicit languages only returns those + English."""
        config = LanguageMixConfig(
            english_ratio=0.9,
            explicit_languages={"de": 0.05, "fr": 0.05},
            others_ratio=0.0,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        languages = mixer.get_languages_for_mixing()

        assert set(languages) == {"en", "de", "fr"}


class TestSampling:
    """Tests for sampling functionality."""

    def test_sample_from_dataframe(self):
        """Can sample correct number of rows from DataFrame."""
        import pandas as pd

        config = LanguageMixConfig(
            english_ratio=0.9,
            explicit_languages={},
            others_ratio=0.1,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        # Create sample data
        df = pd.DataFrame({"text": [f"sample_{i}" for i in range(1000)]})

        # Sample 100 rows
        sampled = mixer.sample_dataframe(df, n_samples=100, seed=42)

        assert len(sampled) == 100
        # Same seed should give same results
        sampled2 = mixer.sample_dataframe(df, n_samples=100, seed=42)
        pd.testing.assert_frame_equal(sampled, sampled2)

    def test_sample_more_than_available_returns_all(self):
        """Sampling more than available returns all rows."""
        import pandas as pd

        config = LanguageMixConfig(
            english_ratio=1.0,
            explicit_languages={},
            others_ratio=0.0,
            datasets=[]
        )
        mixer = LanguageMixer(config)

        df = pd.DataFrame({"text": ["a", "b", "c"]})
        sampled = mixer.sample_dataframe(df, n_samples=100, seed=42)

        assert len(sampled) == 3  # Only 3 rows available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
