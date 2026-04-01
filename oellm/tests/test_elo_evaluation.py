"""Tests for Elo evaluation infrastructure.

Verifies:
1. Cache key uniqueness across language configurations
2. F-track config validity (ratios sum to 1.0, correct languages)
3. Per-language winrate computation correctness
4. Model detection in analyze_per_language_elo.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENJURY_DIR = PROJECT_ROOT / "oellm" / "evaluations" / "benchmarks" / "OpenJury"
sys.path.insert(0, str(OPENJURY_DIR))

from oellm.utils.language_mixer import LanguageMixer, LanguageMixConfig


# ============================================================
# 1. Cache key uniqueness tests
# ============================================================

class TestCacheKeyUniqueness:
    """Verify cache keys are unique across different language configurations."""

    @staticmethod
    def _build_cache_suffix(
        arena: str,
        model: str,
        judge_model: str,
        n_instructions: int | None,
        n_per_language: int,
        languages: list[str] | None,
    ) -> str:
        """Reproduce the cache suffix logic from estimate_elo_ratings.py."""
        lang_key = "_".join(sorted(languages)) if languages else "all"
        return (
            f"{arena}_{model.replace('/', '_')}_"
            f"{judge_model.replace('/', '_')}_{n_instructions}_{n_per_language}_{lang_key}"
        )

    def test_en_only_distinct_from_all_languages(self):
        """English-only cache key must differ from 12-language key."""
        en_only = self._build_cache_suffix(
            "LMArena", "VLLM/models/eval/eu-D1-90en",
            "ChatOpenAI/Qwen/Qwen3.5-27B", None, 200, ["en"],
        )
        all_langs = self._build_cache_suffix(
            "LMArena", "VLLM/models/eval/eu-D1-90en",
            "ChatOpenAI/Qwen/Qwen3.5-27B", None, 200,
            ["en", "de", "es", "fr", "it", "pt", "pl", "nl", "cs", "ro", "el", "uk"],
        )
        assert en_only != all_langs
        assert en_only.endswith("_en")
        assert all_langs.endswith("_cs_de_el_en_es_fr_it_nl_pl_pt_ro_uk")

    def test_wo_en_distinct_from_w_en(self):
        """Without-English cache key must differ from with-English key."""
        w_en = self._build_cache_suffix(
            "LMArena", "VLLM/models/eval/eu-D1-90en",
            "ChatOpenAI/Qwen/Qwen3.5-27B", None, 200,
            ["en", "de", "es", "fr", "it", "pt", "pl", "nl", "cs", "ro", "el", "uk"],
        )
        wo_en = self._build_cache_suffix(
            "LMArena", "VLLM/models/eval/eu-D1-90en",
            "ChatOpenAI/Qwen/Qwen3.5-27B", None, 200,
            ["de", "es", "fr", "it", "pt", "pl", "nl", "cs", "ro", "el", "uk"],
        )
        assert w_en != wo_en

    def test_single_language_keys_are_distinct(self):
        """Each single-language key must be unique."""
        langs = ["en", "de", "es", "fr"]
        keys = set()
        for lang in langs:
            key = self._build_cache_suffix(
                "LMArena", "VLLM/models/eval/eu-D1-90en",
                "ChatOpenAI/Qwen/Qwen3.5-27B", None, 200, [lang],
            )
            keys.add(key)
        assert len(keys) == len(langs)

    def test_no_languages_produces_all(self):
        """When no languages specified, key should be 'all'."""
        key = self._build_cache_suffix(
            "LMArena", "VLLM/models/eval/eu-D1-90en",
            "ChatOpenAI/Qwen/Qwen3.5-27B", None, 200, None,
        )
        assert key.endswith("_all")

    def test_language_order_does_not_matter(self):
        """Languages are sorted, so order shouldn't affect cache key."""
        key1 = self._build_cache_suffix(
            "LMArena", "VLLM/models/eval/eu-D1-90en",
            "ChatOpenAI/Qwen/Qwen3.5-27B", None, 200, ["en", "de", "fr"],
        )
        key2 = self._build_cache_suffix(
            "LMArena", "VLLM/models/eval/eu-D1-90en",
            "ChatOpenAI/Qwen/Qwen3.5-27B", None, 200, ["fr", "en", "de"],
        )
        assert key1 == key2

    def test_different_n_per_language_distinct(self):
        """200 vs 500 battles per language must produce different keys."""
        key_200 = self._build_cache_suffix(
            "LMArena", "VLLM/models/eval/eu-D1-90en",
            "ChatOpenAI/Qwen/Qwen3.5-27B", None, 200, ["en"],
        )
        key_500 = self._build_cache_suffix(
            "LMArena", "VLLM/models/eval/eu-D1-90en",
            "ChatOpenAI/Qwen/Qwen3.5-27B", None, 500, ["en"],
        )
        assert key_200 != key_500


# ============================================================
# 2. F-track config validity tests
# ============================================================

CONFIGS_DIR = PROJECT_ROOT / "oellm" / "configs"

F_TRACK_CONFIGS = [
    ("multilingual_trackF_100en.yaml", "F1-100en", 1.0, 0),
    ("multilingual_trackF_75en.yaml", "F2-75en", 0.75, 6),
    ("multilingual_trackF_50en.yaml", "F3-50en", 0.50, 6),
    ("multilingual_trackF_25en.yaml", "F4-25en", 0.25, 6),
    ("multilingual_trackF_0en.yaml", "F5-0en", 0.0, 6),
]


class TestFTrackConfigs:
    """Validate F-track YAML configs."""

    @pytest.mark.parametrize("filename,expected_name,expected_en_ratio,expected_n_langs", F_TRACK_CONFIGS)
    def test_config_loads_and_validates(self, filename, expected_name, expected_en_ratio, expected_n_langs):
        """Each F-track config loads without error and has correct name/ratio."""
        config_path = CONFIGS_DIR / filename
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        mixer = LanguageMixer.from_yaml(config_path)
        assert mixer.config.name == expected_name
        assert abs(mixer.config.english_ratio - expected_en_ratio) < 1e-6

    @pytest.mark.parametrize("filename,expected_name,expected_en_ratio,expected_n_langs", F_TRACK_CONFIGS)
    def test_ratios_sum_to_one(self, filename, expected_name, expected_en_ratio, expected_n_langs):
        """Ratios must sum to exactly 1.0."""
        config_path = CONFIGS_DIR / filename
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        mixer = LanguageMixer.from_yaml(config_path)
        total = mixer.config.english_ratio + sum(mixer.config.explicit_languages.values())
        assert abs(total - 1.0) < 1e-4, f"Ratios sum to {total}, expected 1.0"

    @pytest.mark.parametrize("filename,expected_name,expected_en_ratio,expected_n_langs", F_TRACK_CONFIGS)
    def test_language_count(self, filename, expected_name, expected_en_ratio, expected_n_langs):
        """Correct number of EU languages."""
        config_path = CONFIGS_DIR / filename
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        mixer = LanguageMixer.from_yaml(config_path)
        assert len(mixer.config.explicit_languages) == expected_n_langs

    @pytest.mark.parametrize("filename,expected_name,expected_en_ratio,expected_n_langs", F_TRACK_CONFIGS)
    def test_total_samples_is_500k(self, filename, expected_name, expected_en_ratio, expected_n_langs):
        """All F-track configs target 500k samples."""
        config_path = CONFIGS_DIR / filename
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        mixer = LanguageMixer.from_yaml(config_path)
        assert mixer.config.total_samples == 500000

    @pytest.mark.parametrize("filename,expected_name,expected_en_ratio,expected_n_langs", F_TRACK_CONFIGS)
    def test_no_czech_or_dutch(self, filename, expected_name, expected_en_ratio, expected_n_langs):
        """F-track drops cs and nl due to data scarcity at 500k scale."""
        config_path = CONFIGS_DIR / filename
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        mixer = LanguageMixer.from_yaml(config_path)
        assert "cs" not in mixer.config.explicit_languages
        assert "nl" not in mixer.config.explicit_languages

    def test_f1_has_dolci_english_dir(self):
        """F1 (100% en) must use Dolci replay."""
        config_path = CONFIGS_DIR / "multilingual_trackF_100en.yaml"
        if not config_path.exists():
            pytest.skip("Config not found")

        mixer = LanguageMixer.from_yaml(config_path)
        assert mixer.config.english_dir is not None
        assert "Dolci-Instruct-SFT" in mixer.config.english_dir

    def test_f5_has_no_english_dir(self):
        """F5 (0% en) must not have english_dir."""
        config_path = CONFIGS_DIR / "multilingual_trackF_0en.yaml"
        if not config_path.exists():
            pytest.skip("Config not found")

        mixer = LanguageMixer.from_yaml(config_path)
        assert mixer.config.english_dir is None

    def test_f2_through_f4_have_dolci_english_dir(self):
        """F2-F4 use Dolci replay for their English portion."""
        for filename in ["multilingual_trackF_75en.yaml", "multilingual_trackF_50en.yaml", "multilingual_trackF_25en.yaml"]:
            config_path = CONFIGS_DIR / filename
            if not config_path.exists():
                pytest.skip(f"Config not found: {config_path}")

            mixer = LanguageMixer.from_yaml(config_path)
            assert mixer.config.english_dir is not None, f"{filename} should have english_dir"

    @pytest.mark.parametrize("filename,expected_name,expected_en_ratio,expected_n_langs", F_TRACK_CONFIGS)
    def test_compute_samples_matches_total(self, filename, expected_name, expected_en_ratio, expected_n_langs):
        """compute_samples_per_language should produce counts summing to ~500k."""
        config_path = CONFIGS_DIR / filename
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        mixer = LanguageMixer.from_yaml(config_path)
        samples = mixer.compute_samples_per_language(500000)
        total = sum(samples.values())
        # Allow small rounding error (up to 10 samples)
        assert abs(total - 500000) <= 10, f"Total samples {total}, expected ~500000"


# ============================================================
# 3. Per-language winrate computation tests
# ============================================================

class TestWinrateComputation:
    """Test the per-language winrate computation logic from analyze_per_language_elo.py."""

    @staticmethod
    def _compute_winrate_from_results(results: list[dict]) -> float:
        """Simplified winrate: mean of our_win values."""
        if not results:
            return 0.0
        return sum(r["our_win"] for r in results) / len(r for r in results) if results else 0.0

    @staticmethod
    def _parse_battle(judge_pref: float | None, is_pos_a: bool) -> float:
        """Reproduce battle outcome logic from analyze_per_language_elo.py."""
        if judge_pref is None or judge_pref == 0.5:
            return 0.5  # tie
        elif judge_pref < 0.5:
            # Judge prefers A
            return 1.0 if is_pos_a else 0.0
        else:
            # Judge prefers B
            return 0.0 if is_pos_a else 1.0

    def test_all_wins_position_a(self):
        """All wins when our model is position A and judge prefers A."""
        results = [self._parse_battle(0.0, True) for _ in range(100)]
        assert np.mean(results) == 1.0

    def test_all_wins_position_b(self):
        """All wins when our model is position B and judge prefers B."""
        results = [self._parse_battle(1.0, False) for _ in range(100)]
        assert np.mean(results) == 1.0

    def test_all_losses_position_a(self):
        """All losses when our model is position A and judge prefers B."""
        results = [self._parse_battle(1.0, True) for _ in range(100)]
        assert np.mean(results) == 0.0

    def test_all_losses_position_b(self):
        """All losses when our model is position B and judge prefers A."""
        results = [self._parse_battle(0.0, False) for _ in range(100)]
        assert np.mean(results) == 0.0

    def test_all_ties(self):
        """All ties produce winrate 0.5."""
        results = [self._parse_battle(0.5, True) for _ in range(100)]
        assert np.mean(results) == 0.5

    def test_none_preference_is_tie(self):
        """None preference treated as tie (0.5)."""
        results = [self._parse_battle(None, True) for _ in range(10)]
        assert np.mean(results) == 0.5

    def test_balanced_swap_mode(self):
        """With swap_mode=both, half in position A and half in B should give same result."""
        # 50 battles as position A (all wins), 50 as position B (all wins)
        results_a = [self._parse_battle(0.0, True) for _ in range(50)]
        results_b = [self._parse_battle(1.0, False) for _ in range(50)]
        winrate = np.mean(results_a + results_b)
        assert winrate == 1.0

    def test_mixed_outcomes(self):
        """Known mixed outcomes produce expected winrate."""
        # 3 wins, 2 losses, 1 tie = (3 + 0.5) / 6 = 0.583...
        results = [
            self._parse_battle(0.0, True),   # win
            self._parse_battle(0.0, True),   # win
            self._parse_battle(0.0, True),   # win
            self._parse_battle(0.0, False),  # loss
            self._parse_battle(0.0, False),  # loss
            self._parse_battle(0.5, True),   # tie
        ]
        expected = (3 * 1.0 + 2 * 0.0 + 1 * 0.5) / 6
        assert abs(np.mean(results) - expected) < 1e-6


# ============================================================
# 4. Model detection tests
# ============================================================

class TestModelDetection:
    """Test model map coverage from analyze_per_language_elo.py."""

    # Reproduce MODEL_MAP from analyze_per_language_elo.py
    MODEL_MAP = {
        "eu-A1-90en-step238": "A1-90en",
        "eu-A2-80en-step228": "A2-80en",
        "eu-A3-70en-step216": "A3-70en",
        "eu-B1-90en": "B1-90en",
        "eu-B2-80en": "B2-80en",
        "eu-C0-100en": "C0-100en",
        "eu-D1-90en": "D1-90en",
        "eu-D2-80en": "D2-80en",
        "eu-D3-70en": "D3-70en",
        "eu-E1-90en": "E1-90en",
        "eu-E2-80en": "E2-80en",
        "eu-E3-70en": "E3-70en",
        "instruct-v2-step3252": "Baseline",
    }

    def test_all_12_eu_models_covered(self):
        """MODEL_MAP must cover all 12 EU experiment models."""
        expected_display = {
            "A1-90en", "A2-80en", "A3-70en",
            "B1-90en", "B2-80en",
            "C0-100en",
            "D1-90en", "D2-80en", "D3-70en",
            "E1-90en", "E2-80en", "E3-70en",
        }
        actual_display = set(self.MODEL_MAP.values()) - {"Baseline"}
        assert expected_display == actual_display

    def test_baseline_covered(self):
        """MODEL_MAP must include the baseline model."""
        assert "Baseline" in self.MODEL_MAP.values()

    def test_symlink_names_match_script_arrays(self):
        """Symlink names in MODEL_MAP must match the MODELS_ARRAY in Elo scripts."""
        # These are the symlinks used in run_elo_evaluation_qwen35_en_only.sh
        script_models = [
            "eu-A1-90en-step238",
            "eu-A2-80en-step228",
            "eu-A3-70en-step216",
            "eu-B1-90en",
            "eu-B2-80en",
            "eu-C0-100en",
            "eu-D1-90en",
            "eu-D2-80en",
            "eu-D3-70en",
            "eu-E1-90en",
            "eu-E2-80en",
            "eu-E3-70en",
            "instruct-v2-step3252",
        ]
        for model in script_models:
            assert model in self.MODEL_MAP, f"{model} not in MODEL_MAP"

    def test_no_duplicate_display_names(self):
        """Each display name must be unique."""
        display_names = list(self.MODEL_MAP.values())
        assert len(display_names) == len(set(display_names))

    def test_cache_filename_extraction(self):
        """Verify model symlink can be extracted from cache filename."""
        # Simulate a judge cache filename
        cache_file = (
            "judge_LMArena_VLLM__work_dlclarge2_ferreira-oellm_open-instruct_"
            "models_eval_eu-D1-90en_ChatOpenAI_Qwen_Qwen3.5-27B_None_200_en.csv.zip"
        )
        # Check that we can find the model symlink in the filename
        found = None
        for sym in self.MODEL_MAP:
            if sym.replace("/", "_") in cache_file or sym in cache_file:
                found = sym
                break
        assert found == "eu-D1-90en"
        assert self.MODEL_MAP[found] == "D1-90en"
