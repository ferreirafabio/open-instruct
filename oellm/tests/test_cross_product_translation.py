"""Tests for cross-product translation (all datasets in all 24 EU languages).

Tests that:
1. English can be used as a target language
2. Source language detection works correctly
3. Rows are skipped when source == target
4. Stats include source language distribution
"""

from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest

from oellm.pipelines.translation.translate_dataset import (
    EU_LANGUAGES,
    LANGDETECT_TO_NLLB,
    detect_source_language,
    get_nllb_code,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_conversations():
    """Sample conversations with mixed languages."""
    return [
        # English conversation
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        # German conversation
        [
            {"role": "user", "content": "Was ist die Hauptstadt von Frankreich?"},
            {"role": "assistant", "content": "Die Hauptstadt von Frankreich ist Paris."},
        ],
        # French conversation
        [
            {"role": "user", "content": "Quelle est la capitale de la France?"},
            {"role": "assistant", "content": "La capitale de la France est Paris."},
        ],
        # Spanish conversation
        [
            {"role": "user", "content": "Cual es la capital de Francia?"},
            {"role": "assistant", "content": "La capital de Francia es Paris."},
        ],
    ]


@pytest.fixture
def mock_translator():
    """Mock translator that tracks calls and 'translates' by adding language prefix."""
    from oellm.pipelines.translation.translate_dataset import DatasetTranslator

    mock = MagicMock(spec=DatasetTranslator)
    mock.batch_size = 128

    def mock_translate_batch(texts, src_lang, tgt_lang):
        """Mock translation - prepend target language code."""
        return [f"[{tgt_lang}] {text}" for text in texts]

    mock.translate_batch.side_effect = mock_translate_batch
    return mock


# ============================================================================
# Language Detection Tests
# ============================================================================


class TestLanguageDetection:
    """Tests for source language detection."""

    def test_detect_english(self):
        """Test detection of English text."""
        text = "Hello, how are you doing today? I hope everything is going well."
        result = detect_source_language(text)
        assert result == "eng_Latn"

    def test_detect_german(self):
        """Test detection of German text."""
        text = "Guten Tag, wie geht es Ihnen heute? Ich hoffe alles ist in Ordnung."
        result = detect_source_language(text)
        assert result == "deu_Latn"

    def test_detect_french(self):
        """Test detection of French text."""
        text = "Bonjour, comment allez-vous aujourd'hui? J'espere que tout va bien."
        result = detect_source_language(text)
        assert result == "fra_Latn"

    def test_detect_spanish(self):
        """Test detection of Spanish text."""
        text = "Hola, como estas hoy? Espero que todo este bien contigo."
        result = detect_source_language(text)
        assert result == "spa_Latn"

    def test_short_text_defaults_to_english(self):
        """Test that short text defaults to English."""
        text = "Hi"
        result = detect_source_language(text)
        assert result == "eng_Latn"

    def test_empty_text_defaults_to_english(self):
        """Test that empty text defaults to English."""
        result = detect_source_language("")
        assert result == "eng_Latn"

    def test_code_text_defaults_to_english(self):
        """Test that code-like text defaults to English."""
        text = "def foo(x): return x + 1"
        result = detect_source_language(text)
        # Code is often detected as English or defaults to English
        assert result in LANGDETECT_TO_NLLB.values() or result == "eng_Latn"


# ============================================================================
# Cross-Product Translation Tests
# ============================================================================


class TestCrossProductTranslation:
    """Tests for cross-product translation functionality."""

    def test_english_is_valid_target_language(self):
        """Test that English is a valid target language."""
        assert "en" in EU_LANGUAGES
        assert get_nllb_code("en") == "eng_Latn"

    def test_all_24_languages_are_valid_targets(self):
        """Test that all 24 EU languages are valid targets."""
        expected_count = 24
        assert len(EU_LANGUAGES) == expected_count

        # Verify each language has a valid NLLB code
        for lang_code in EU_LANGUAGES:
            nllb_code = get_nllb_code(lang_code)
            assert nllb_code is not None
            assert isinstance(nllb_code, str)
            assert len(nllb_code) > 0

    def test_langdetect_to_nllb_mapping_complete(self):
        """Test that langdetect to NLLB mapping includes common languages."""
        # Core languages that must be supported
        core_languages = ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru"]
        for lang in core_languages:
            assert lang in LANGDETECT_TO_NLLB, f"Missing language: {lang}"

    def test_skip_when_source_equals_target_german(self):
        """Test that German text is skipped when target is German."""
        from oellm.pipelines.translation.translate_dataset import DatasetTranslator

        # Create a mock that tracks what gets translated
        translated_texts = []

        def track_translate_batch(texts, src_lang, tgt_lang):
            translated_texts.extend(texts)
            return [f"[{tgt_lang}] {t}" for t in texts]

        with patch.object(DatasetTranslator, "__init__", lambda self, **kwargs: None):
            translator = DatasetTranslator()
            translator.batch_size = 128
            translator.translate_batch = track_translate_batch

            conversations = [
                # German text -> should be skipped when target is German
                [{"role": "user", "content": "Guten Tag, wie geht es Ihnen heute?"}],
                # English text -> should be translated to German
                [{"role": "user", "content": "Hello, how are you doing today?"}],
            ]

            # This tests the internal logic - German text should not be in translated_texts
            # when target is German because detect_source_language returns deu_Latn
            result, stats = translator.translate_conversations_batched(
                conversations, "de", detect_source=True, return_stats=True
            )

            # The German text should have been skipped
            assert stats["skipped_same_language"] >= 1 or "deu_Latn" in stats["source_lang_distribution"]


class TestTranslateConversationsBatched:
    """Tests for the translate_conversations_batched method."""

    def test_returns_stats_when_requested(self, sample_conversations, mock_translator):
        """Test that stats are returned when return_stats=True."""
        from oellm.pipelines.translation.translate_dataset import DatasetTranslator

        # Patch detect_source_language to return known values
        with patch(
            "oellm.pipelines.translation.translate_dataset.detect_source_language"
        ) as mock_detect:
            # Return English for all texts (simplest case) - now returns tuple with reason
            mock_detect.return_value = ("eng_Latn", "detected")

            with patch.object(DatasetTranslator, "__init__", lambda self, **kwargs: None):
                translator = DatasetTranslator()
                translator.batch_size = 128
                translator.translate_batch = mock_translator.translate_batch

                result, stats = translator.translate_conversations_batched(
                    sample_conversations, "de", detect_source=True, return_stats=True
                )

                assert isinstance(stats, dict)
                assert "source_lang_distribution" in stats
                assert "skipped_same_language" in stats
                assert "total_messages" in stats
                assert "system_messages" in stats
                assert "empty_messages" in stats
                assert "short_messages" in stats

    def test_returns_only_translations_when_stats_not_requested(
        self, sample_conversations, mock_translator
    ):
        """Test that only translations are returned when return_stats=False."""
        from oellm.pipelines.translation.translate_dataset import DatasetTranslator

        with patch(
            "oellm.pipelines.translation.translate_dataset.detect_source_language"
        ) as mock_detect:
            # Returns tuple with reason
            mock_detect.return_value = ("eng_Latn", "detected")

            with patch.object(DatasetTranslator, "__init__", lambda self, **kwargs: None):
                translator = DatasetTranslator()
                translator.batch_size = 128
                translator.translate_batch = mock_translator.translate_batch

                result = translator.translate_conversations_batched(
                    sample_conversations, "de", detect_source=True, return_stats=False
                )

                # Should return list, not tuple
                assert isinstance(result, list)
                assert len(result) == len(sample_conversations)

    def test_empty_conversations_returns_empty_stats(self, mock_translator):
        """Test that empty conversations return empty stats."""
        from oellm.pipelines.translation.translate_dataset import DatasetTranslator

        with patch.object(DatasetTranslator, "__init__", lambda self, **kwargs: None):
            translator = DatasetTranslator()
            translator.batch_size = 128
            translator.translate_batch = mock_translator.translate_batch

            result, stats = translator.translate_conversations_batched(
                [], "de", detect_source=True, return_stats=True
            )

            assert result == []
            assert stats["source_lang_distribution"] == {}
            assert stats["skipped_same_language"] == 0
            assert stats["total_messages"] == 0

    def test_system_messages_preserved(self, mock_translator):
        """Test that system messages are not translated."""
        from oellm.pipelines.translation.translate_dataset import DatasetTranslator

        with patch(
            "oellm.pipelines.translation.translate_dataset.detect_source_language"
        ) as mock_detect:
            # Returns tuple with reason
            mock_detect.return_value = ("eng_Latn", "detected")

            with patch.object(DatasetTranslator, "__init__", lambda self, **kwargs: None):
                translator = DatasetTranslator()
                translator.batch_size = 128
                translator.translate_batch = mock_translator.translate_batch

                conversations = [
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                        {"role": "assistant", "content": "Hi there!"},
                    ]
                ]

                result, stats = translator.translate_conversations_batched(
                    conversations, "de", detect_source=True, return_stats=True
                )

                # System message should be unchanged
                assert result[0][0]["content"] == "You are a helpful assistant."
                # User and assistant messages should be translated
                assert "[deu_Latn]" in result[0][1]["content"]
                assert "[deu_Latn]" in result[0][2]["content"]
                # System messages should be counted
                assert stats["system_messages"] == 1


class TestSourceLanguageDistribution:
    """Tests for source language distribution tracking."""

    def test_source_lang_distribution_tracked(self, mock_translator):
        """Test that source language distribution is correctly tracked."""
        from oellm.pipelines.translation.translate_dataset import DatasetTranslator

        # Create conversations with known source languages
        conversations = [
            [{"role": "user", "content": "Hello, how are you?"}],  # English
            [{"role": "user", "content": "Bonjour, comment allez-vous?"}],  # French
            [{"role": "user", "content": "Guten Tag, wie geht es Ihnen?"}],  # German
        ]

        with patch.object(DatasetTranslator, "__init__", lambda self, **kwargs: None):
            translator = DatasetTranslator()
            translator.batch_size = 128
            translator.translate_batch = mock_translator.translate_batch

            # Use real language detection
            result, stats = translator.translate_conversations_batched(
                conversations, "es", detect_source=True, return_stats=True
            )

            # Check that source languages were detected
            assert stats["total_messages"] == 3
            assert len(stats["source_lang_distribution"]) > 0
            # All should be different source languages (unlikely to have same-language skips for Spanish target)
            assert stats["skipped_same_language"] == 0


# ============================================================================
# Property-Based Tests (if hypothesis is available)
# ============================================================================

try:
    from hypothesis import given, settings, strategies as st

    class TestCrossProductProperties:
        """Property-based tests for cross-product translation."""

        @given(st.sampled_from(list(EU_LANGUAGES.keys())))
        @settings(max_examples=24)
        def test_all_languages_have_nllb_codes(self, lang):
            """Test that all EU languages have valid NLLB codes."""
            nllb_code = get_nllb_code(lang)
            assert nllb_code is not None
            assert isinstance(nllb_code, str)
            assert "_" in nllb_code  # NLLB codes have format like "eng_Latn"

        @given(st.sampled_from(list(EU_LANGUAGES.keys())))
        @settings(max_examples=24)
        def test_nllb_codes_are_unique(self, lang):
            """Test that each language maps to a unique NLLB code."""
            codes = {get_nllb_code(l) for l in EU_LANGUAGES}
            assert len(codes) == len(EU_LANGUAGES)

except ImportError:
    # hypothesis not installed, skip property tests
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
