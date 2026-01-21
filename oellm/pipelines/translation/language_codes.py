"""
Language code mappings for 24 EU official languages.

Provides mappings between:
- ISO 639-1 codes (2-letter)
- NLLB-200 language codes (used by facebook/nllb-200-3.3B)
- Full language names
"""

# All 24 official EU languages with metadata
EU_LANGUAGES = {
    "bg": {"name": "Bulgarian", "nllb": "bul_Cyrl", "script": "Cyrillic", "countries": ["Bulgaria"]},
    "hr": {"name": "Croatian", "nllb": "hrv_Latn", "script": "Latin", "countries": ["Croatia"]},
    "cs": {"name": "Czech", "nllb": "ces_Latn", "script": "Latin", "countries": ["Czech Republic"]},
    "da": {"name": "Danish", "nllb": "dan_Latn", "script": "Latin", "countries": ["Denmark"]},
    "nl": {"name": "Dutch", "nllb": "nld_Latn", "script": "Latin", "countries": ["Netherlands", "Belgium"]},
    "en": {"name": "English", "nllb": "eng_Latn", "script": "Latin", "countries": ["Ireland", "Malta"]},
    "et": {"name": "Estonian", "nllb": "est_Latn", "script": "Latin", "countries": ["Estonia"]},
    "fi": {"name": "Finnish", "nllb": "fin_Latn", "script": "Latin", "countries": ["Finland"]},
    "fr": {"name": "French", "nllb": "fra_Latn", "script": "Latin", "countries": ["France", "Belgium", "Luxembourg"]},
    "de": {"name": "German", "nllb": "deu_Latn", "script": "Latin", "countries": ["Germany", "Austria", "Belgium", "Luxembourg"]},
    "el": {"name": "Greek", "nllb": "ell_Grek", "script": "Greek", "countries": ["Greece", "Cyprus"]},
    "hu": {"name": "Hungarian", "nllb": "hun_Latn", "script": "Latin", "countries": ["Hungary"]},
    "ga": {"name": "Irish", "nllb": "gle_Latn", "script": "Latin", "countries": ["Ireland"]},
    "it": {"name": "Italian", "nllb": "ita_Latn", "script": "Latin", "countries": ["Italy"]},
    "lv": {"name": "Latvian", "nllb": "lvs_Latn", "script": "Latin", "countries": ["Latvia"]},
    "lt": {"name": "Lithuanian", "nllb": "lit_Latn", "script": "Latin", "countries": ["Lithuania"]},
    "mt": {"name": "Maltese", "nllb": "mlt_Latn", "script": "Latin", "countries": ["Malta"]},
    "pl": {"name": "Polish", "nllb": "pol_Latn", "script": "Latin", "countries": ["Poland"]},
    "pt": {"name": "Portuguese", "nllb": "por_Latn", "script": "Latin", "countries": ["Portugal"]},
    "ro": {"name": "Romanian", "nllb": "ron_Latn", "script": "Latin", "countries": ["Romania"]},
    "sk": {"name": "Slovak", "nllb": "slk_Latn", "script": "Latin", "countries": ["Slovakia"]},
    "sl": {"name": "Slovenian", "nllb": "slv_Latn", "script": "Latin", "countries": ["Slovenia"]},
    "es": {"name": "Spanish", "nllb": "spa_Latn", "script": "Latin", "countries": ["Spain"]},
    "sv": {"name": "Swedish", "nllb": "swe_Latn", "script": "Latin", "countries": ["Sweden"]},
}

# Quick lookup: ISO code -> NLLB code
NLLB_CODES = {iso: info["nllb"] for iso, info in EU_LANGUAGES.items()}

# Reverse lookup: NLLB code -> ISO code
ISO_TO_NLLB = {info["nllb"]: iso for iso, info in EU_LANGUAGES.items()}

# Languages covered by m-arena-hard-EU benchmark (for evaluation)
BENCHMARK_LANGUAGES = {"cs", "de", "el", "en", "es", "fr", "it", "nl", "pl", "pt", "ro"}
# Note: uk (Ukrainian) is also in the benchmark but not an EU official language

# Languages NOT covered by m-arena-hard-EU (12 of 24)
MISSING_FROM_BENCHMARK = {"bg", "da", "et", "fi", "ga", "hr", "hu", "lt", "lv", "mt", "sk", "sl", "sv"}

# Language groupings by expected tokenizer fertility
FERTILITY_GROUPS = {
    "excellent": {"en", "de", "fr", "nl", "es", "it", "pt", "da", "sv"},  # 1.0-1.5x
    "good": {"ro", "hr", "hu", "pl", "sl", "et", "fi", "ga"},  # 1.5-2.0x
    "moderate": {"sk", "lt", "lv", "cs"},  # 2.0-2.5x
    "challenging": {"mt", "bg", "el"},  # 2.5-4.0x (different scripts or low-resource)
}


def get_nllb_code(iso_code: str) -> str:
    """Get NLLB language code from ISO 639-1 code."""
    if iso_code not in NLLB_CODES:
        raise ValueError(f"Unknown language code: {iso_code}. Valid codes: {list(NLLB_CODES.keys())}")
    return NLLB_CODES[iso_code]


def get_language_name(iso_code: str) -> str:
    """Get full language name from ISO 639-1 code."""
    if iso_code not in EU_LANGUAGES:
        raise ValueError(f"Unknown language code: {iso_code}")
    return EU_LANGUAGES[iso_code]["name"]


def get_non_english_languages() -> list[str]:
    """Get list of non-English EU language codes."""
    return [lang for lang in EU_LANGUAGES.keys() if lang != "en"]


def get_evaluation_languages() -> list[str]:
    """Get list of languages covered by m-arena-hard-EU benchmark."""
    return sorted(BENCHMARK_LANGUAGES)
