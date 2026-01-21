"""Language mixer for multilingual dataset mixing.

Computes per-language sample counts based on configurable ratios.
Supports explicit per-language ratios and "others" pool distribution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


# All 23 EU official languages (excluding English)
EU_LANGUAGES = [
    "bg",  # Bulgarian
    "cs",  # Czech
    "da",  # Danish
    "de",  # German
    "el",  # Greek
    "es",  # Spanish
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "ga",  # Irish
    "hr",  # Croatian
    "hu",  # Hungarian
    "it",  # Italian
    "lt",  # Lithuanian
    "lv",  # Latvian
    "mt",  # Maltese
    "nl",  # Dutch
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "sk",  # Slovak
    "sl",  # Slovenian
    "sv",  # Swedish
]


@dataclass
class LanguageMixConfig:
    """Configuration for language mixing.

    Attributes:
        english_ratio: Ratio of English samples (e.g., 0.9 for 90%)
        explicit_languages: Dict of language code -> ratio (e.g., {"de": 0.02})
        others_ratio: Ratio to distribute among non-explicit EU languages
        datasets: List of dataset filenames to include
        seed: Random seed for reproducibility
        translation_model: Translation model used (for metadata)
    """

    english_ratio: float
    explicit_languages: dict[str, float] = field(default_factory=dict)
    others_ratio: float = 0.0
    datasets: list[str] = field(default_factory=list)
    seed: int = 42
    translation_model: Optional[str] = None


class LanguageMixer:
    """Computes per-language sample counts based on config.

    Example:
        >>> config = LanguageMixConfig(
        ...     english_ratio=0.9,
        ...     explicit_languages={"de": 0.05, "fr": 0.03},
        ...     others_ratio=0.02,
        ...     datasets=["dataset1.parquet"]
        ... )
        >>> mixer = LanguageMixer(config)
        >>> samples = mixer.compute_samples_per_language(100000)
        >>> print(samples)
        {"en": 90000, "de": 5000, "fr": 3000, "bg": 95, "cs": 95, ...}
    """

    def __init__(self, config: LanguageMixConfig):
        """Initialize with configuration.

        Args:
            config: Language mix configuration

        Raises:
            ValueError: If ratios don't sum to 1.0 or invalid language codes
        """
        self.config = config
        self._validate()

    def _validate(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If validation fails
        """
        # Check for negative ratios
        if self.config.english_ratio < 0:
            raise ValueError("english_ratio cannot be negative")
        if self.config.others_ratio < 0:
            raise ValueError("others_ratio cannot be negative")
        for lang, ratio in self.config.explicit_languages.items():
            if ratio < 0:
                raise ValueError(f"Ratio for {lang} cannot be negative")

        # Check that explicit languages are valid EU languages
        for lang in self.config.explicit_languages:
            if lang == "en":
                raise ValueError(
                    "English should not be in explicit_languages; "
                    "use english_ratio instead"
                )
            if lang not in EU_LANGUAGES:
                raise ValueError(
                    f"'{lang}' is not a valid EU language. "
                    f"Valid languages: {EU_LANGUAGES}"
                )

        # Check that ratios sum to 1.0
        total = self.config.english_ratio
        total += sum(self.config.explicit_languages.values())
        total += self.config.others_ratio

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {total:.6f}. "
                f"english_ratio={self.config.english_ratio}, "
                f"explicit_sum={sum(self.config.explicit_languages.values())}, "
                f"others_ratio={self.config.others_ratio}"
            )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "LanguageMixer":
        """Load configuration from a YAML file.

        YAML format:
            english_ratio: 0.9
            languages:
              de: 0.05
              fr: 0.03
              others: 0.02
            datasets:
              - dataset1.parquet
              - dataset2.parquet
            seed: 42  # optional

        Args:
            config_path: Path to YAML config file

        Returns:
            LanguageMixer instance
        """
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        # Parse languages section
        languages = raw_config.get("languages", {})
        if isinstance(languages, list):
            raise TypeError("languages must be a dict, not a list")

        # Separate "others" from explicit languages
        others_ratio = languages.pop("others", 0.0) if isinstance(languages, dict) else 0.0
        explicit_languages = languages if isinstance(languages, dict) else {}

        config = LanguageMixConfig(
            english_ratio=raw_config["english_ratio"],
            explicit_languages=explicit_languages,
            others_ratio=others_ratio,
            datasets=raw_config.get("datasets", []),
            seed=raw_config.get("seed", 42),
            translation_model=raw_config.get("translation_model"),
        )

        return cls(config)

    def compute_samples_per_language(self, total_samples: int) -> dict[str, int]:
        """Compute number of samples for each language.

        Args:
            total_samples: Total number of samples to distribute

        Returns:
            Dict mapping language code to sample count
        """
        result: dict[str, int] = {}

        # English samples
        result["en"] = int(total_samples * self.config.english_ratio)

        # Explicit language samples
        for lang, ratio in self.config.explicit_languages.items():
            result[lang] = int(total_samples * ratio)

        # Distribute "others" among remaining EU languages
        if self.config.others_ratio > 0:
            others = [
                lang for lang in EU_LANGUAGES
                if lang not in self.config.explicit_languages
            ]

            if others:
                per_other = int(total_samples * self.config.others_ratio / len(others))
                for lang in others:
                    result[lang] = per_other

        return result

    def get_languages_for_mixing(self) -> list[str]:
        """Get list of languages that will have samples.

        Returns:
            List of language codes with non-zero ratios
        """
        languages = ["en"]

        # Add explicit languages
        languages.extend(self.config.explicit_languages.keys())

        # Add "others" languages if ratio > 0
        if self.config.others_ratio > 0:
            others = [
                lang for lang in EU_LANGUAGES
                if lang not in self.config.explicit_languages
            ]
            languages.extend(others)

        return languages

    def sample_dataframe(
        self,
        df: pd.DataFrame,
        n_samples: int,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Sample rows from a DataFrame.

        Args:
            df: DataFrame to sample from
            n_samples: Number of samples to take
            seed: Random seed (uses config seed if not provided)

        Returns:
            Sampled DataFrame
        """
        seed = seed if seed is not None else self.config.seed

        if len(df) <= n_samples:
            return df

        return df.sample(n=n_samples, random_state=seed)

    def get_config_summary(self) -> str:
        """Get a human-readable summary of the configuration.

        Returns:
            Summary string
        """
        lines = [
            f"Language Mix Configuration:",
            f"  English: {self.config.english_ratio * 100:.1f}%",
        ]

        for lang, ratio in sorted(self.config.explicit_languages.items()):
            lines.append(f"  {lang.upper()}: {ratio * 100:.1f}%")

        if self.config.others_ratio > 0:
            others_count = len(EU_LANGUAGES) - len(self.config.explicit_languages)
            per_other = self.config.others_ratio / others_count * 100
            lines.append(
                f"  Others ({others_count} languages): "
                f"{self.config.others_ratio * 100:.1f}% total "
                f"({per_other:.2f}% each)"
            )

        lines.append(f"  Datasets: {len(self.config.datasets)}")
        lines.append(f"  Seed: {self.config.seed}")

        return "\n".join(lines)
