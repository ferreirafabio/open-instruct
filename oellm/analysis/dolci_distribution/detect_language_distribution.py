"""Detect language distribution in Dolci tokenized datasets.

Analyzes the multilingual composition of allenai/Dolci-Instruct-SFT and
allenai/Dolci-Think-SFT-7B by detecting languages of user messages using FastText.

Supports multi-node parallelism via sharding: each SLURM array task processes
a slice of the dataset, then a merge step combines results and generates plots.
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

DATASET_NAMES = {
    "instruct": "allenai/Dolci-Instruct-SFT",
    "think": "allenai/Dolci-Think-SFT-7B",
}

# Top N languages to show in plots; rest grouped as "other"
TOP_N_LANGUAGES = 50

# ISO 639-1 code to full language name (for plot labels)
LANG_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "az": "Azerbaijani",
    "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fr": "French", "ga": "Irish", "gl": "Galician", "gu": "Gujarati",
    "ha": "Hausa", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "jv": "Javanese", "ka": "Georgian",
    "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean",
    "la": "Latin", "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy",
    "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi",
    "ms": "Malay", "mt": "Maltese", "my": "Burmese", "ne": "Nepali",
    "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "si": "Sinhala",
    "sk": "Slovak", "sl": "Slovenian", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "tl": "Tagalog",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "yo": "Yoruba", "zh": "Chinese", "ky": "Kyrgyz",
    # Additional codes found in Dolci datasets
    "ceb": "Cebuano", "ps": "Pashto", "war": "Waray", "eo": "Esperanto",
    "nds": "Low German", "br": "Breton", "sd": "Sindhi", "kv": "Komi",
    "sa": "Sanskrit", "su": "Sundanese", "kw": "Cornish", "ckb": "Central Kurdish",
    "wuu": "Wu Chinese", "tt": "Tatar", "vep": "Veps", "ilo": "Ilocano",
    "qu": "Quechua", "jbo": "Lojban", "rm": "Romansh", "fy": "Western Frisian",
    "io": "Ido", "lmo": "Lombard", "sh": "Serbo-Croatian", "ht": "Haitian Creole",
    "lo": "Lao", "oc": "Occitan", "gv": "Manx", "gd": "Scottish Gaelic",
    "pnb": "Western Punjabi", "pfl": "Palatinate German", "arz": "Egyptian Arabic",
    "cbk": "Chavacano", "yue": "Cantonese", "lb": "Luxembourgish",
    "ast": "Asturian", "als": "Alemannic German", "ia": "Interlingua",
    "xal": "Kalmyk", "pms": "Piedmontese", "wa": "Walloon", "eml": "Emilian-Romagnol",
    "gn": "Guarani", "ku": "Kurdish", "nah": "Nahuatl", "bar": "Bavarian",
    "vec": "Venetian", "hsb": "Upper Sorbian", "nn": "Norwegian Nynorsk",
    "pam": "Pampanga", "sc": "Sardinian", "gom": "Goan Konkani", "ie": "Interlingue",
    "scn": "Sicilian", "li": "Limburgish", "ug": "Uyghur", "co": "Corsican",
    "tk": "Turkmen", "lrc": "Northern Luri", "an": "Aragonese", "ba": "Bashkir",
    "sah": "Yakut", "vo": "Volapuk", "bo": "Tibetan", "bcl": "Central Bikol",
    "ce": "Chechen", "nap": "Neapolitan", "as": "Assamese", "min": "Minangkabau",
    "yi": "Yiddish", "new": "Newari", "mwl": "Mirandese", "tyv": "Tuvan",
    "mrj": "Western Mari", "diq": "Zazaki", "dv": "Maldivian", "sco": "Scots",
    "cv": "Chuvash", "mhr": "Eastern Mari", "lez": "Lezgian", "krc": "Karachay-Balkar",
    "vls": "West Flemish", "azb": "South Azerbaijani", "or": "Odia", "dsb": "Lower Sorbian",
    "bh": "Bihari",
    "unknown": "unknown",
}


def lang_label(code: str) -> str:
    """Convert ISO 639-1 code to 'Name (code)' for plot labels."""
    name = LANG_NAMES.get(code, code)
    if name == code:
        return code
    return f"{name} ({code})"


def extract_user_text(example: dict) -> dict:
    """Concatenate all user messages for language detection."""
    user_texts = [
        m["content"]
        for m in example["messages"]
        if m["role"] == "user" and m.get("content")
    ]
    return {"user_text": " ".join(user_texts)}


def detect_languages_batch(batch: dict) -> dict:
    """Detect language for a batch of texts using FastText."""
    from fast_langdetect import detect

    langs = []
    scores = []
    for text in batch["user_text"]:
        if not text or len(text.strip()) < 20:
            langs.append("unknown")
            scores.append(0.0)
        else:
            try:
                # model="lite" uses bundled lid.176.ftz (938KB, no download).
                # "auto" (default) tries the 125MB model first, which fails
                # in multiprocess workers that all try to download it at once.
                results = detect(text, model="lite")
                top = results[0]
                langs.append(top["lang"])
                scores.append(top["score"])
            except Exception:
                langs.append("unknown")
                scores.append(0.0)
    return {"detected_lang": langs, "detection_score": scores}


def aggregate_results(ds: Dataset) -> dict:
    """Aggregate language detection results into summary statistics."""
    langs = ds["detected_lang"]
    scores = ds["detection_score"]

    # Overall language counts
    lang_counts = Counter(langs)
    total = len(langs)
    lang_percentages = {
        lang: round(count / total * 100, 4) for lang, count in lang_counts.most_common()
    }

    # Confidence statistics
    scores_arr = np.array(scores)
    low_conf_mask = scores_arr < 0.5
    low_conf_langs = [lang for lang, s in zip(langs, scores) if s < 0.5]
    low_conf_counts = Counter(low_conf_langs)

    confidence_stats = {
        "mean": float(np.mean(scores_arr)),
        "median": float(np.median(scores_arr)),
        "std": float(np.std(scores_arr)),
        "min": float(np.min(scores_arr)),
        "max": float(np.max(scores_arr)),
        "low_confidence_count": int(low_conf_mask.sum()),
        "low_confidence_percentage": round(float(low_conf_mask.mean()) * 100, 4),
        "low_confidence_languages": dict(low_conf_counts.most_common()),
    }

    # Text length statistics
    text_lengths = [len(t) for t in ds["user_text"]]
    lengths_arr = np.array(text_lengths)
    text_length_stats = {
        "mean": float(np.mean(lengths_arr)),
        "median": float(np.median(lengths_arr)),
        "std": float(np.std(lengths_arr)),
    }

    # Per-source_dataset breakdown
    per_source = {}
    if "source_dataset" in ds.column_names:
        sources = ds["source_dataset"]
        for src, lang in zip(sources, langs):
            if src not in per_source:
                per_source[src] = Counter()
            per_source[src][lang] += 1
        # Convert to sorted dicts
        per_source = {
            src: dict(counts.most_common()) for src, counts in sorted(per_source.items())
        }

    # Per-domain breakdown
    per_domain = {}
    if "domain" in ds.column_names:
        domains = ds["domain"]
        for dom, lang in zip(domains, langs):
            if dom not in per_domain:
                per_domain[dom] = Counter()
            per_domain[dom][lang] += 1
        per_domain = {
            dom: dict(counts.most_common()) for dom, counts in sorted(per_domain.items())
        }

    return {
        "total_samples": total,
        "language_counts": dict(lang_counts.most_common()),
        "language_percentages": lang_percentages,
        "confidence_stats": confidence_stats,
        "text_length_stats": text_length_stats,
        "per_source_dataset": per_source,
        "per_domain": per_domain,
    }


def merge_shard_results(shard_results: list[dict]) -> dict:
    """Merge results from multiple shards into a single aggregated result.

    Note: median and std for confidence scores and text lengths are approximate
    (taken from the first shard), since computing exact values requires raw data.
    """
    if not shard_results:
        raise ValueError("Cannot merge an empty list of shard results")

    total_samples = sum(r["total_samples"] for r in shard_results)
    if total_samples == 0:
        raise ValueError("Cannot merge shards with zero total samples")

    # Merge language counts
    lang_counts: Counter = Counter()
    for r in shard_results:
        lang_counts.update(r["language_counts"])
    lang_percentages = {
        lang: round(count / total_samples * 100, 4)
        for lang, count in lang_counts.most_common()
    }

    # Merge confidence stats (weighted by sample count)
    # For mean: weighted average. For median/std: approximate from per-shard stats.
    total_low_conf = sum(r["confidence_stats"]["low_confidence_count"] for r in shard_results)
    weighted_mean = sum(
        r["confidence_stats"]["mean"] * r["total_samples"] for r in shard_results
    ) / total_samples
    all_mins = [r["confidence_stats"]["min"] for r in shard_results]
    all_maxs = [r["confidence_stats"]["max"] for r in shard_results]

    # Merge low-confidence language counts
    low_conf_langs: Counter = Counter()
    for r in shard_results:
        low_conf_langs.update(r["confidence_stats"]["low_confidence_languages"])

    confidence_stats = {
        "mean": float(weighted_mean),
        "median": float(shard_results[0]["confidence_stats"]["median"]),  # approximate
        "std": float(shard_results[0]["confidence_stats"]["std"]),  # approximate
        "min": float(min(all_mins)),
        "max": float(max(all_maxs)),
        "low_confidence_count": total_low_conf,
        "low_confidence_percentage": round(total_low_conf / total_samples * 100, 4),
        "low_confidence_languages": dict(low_conf_langs.most_common()),
    }

    # Merge text length stats (weighted mean, approximate std)
    weighted_text_mean = sum(
        r["text_length_stats"]["mean"] * r["total_samples"] for r in shard_results
    ) / total_samples
    text_length_stats = {
        "mean": float(weighted_text_mean),
        "median": float(shard_results[0]["text_length_stats"]["median"]),
        "std": float(shard_results[0]["text_length_stats"]["std"]),
    }

    # Merge per-source_dataset
    per_source: dict[str, Counter] = {}
    for r in shard_results:
        for src, counts in r.get("per_source_dataset", {}).items():
            if src not in per_source:
                per_source[src] = Counter()
            per_source[src].update(counts)
    per_source_sorted = {
        src: dict(counts.most_common()) for src, counts in sorted(per_source.items())
    }

    # Merge per-domain
    per_domain: dict[str, Counter] = {}
    for r in shard_results:
        for dom, counts in r.get("per_domain", {}).items():
            if dom not in per_domain:
                per_domain[dom] = Counter()
            per_domain[dom].update(counts)
    per_domain_sorted = {
        dom: dict(counts.most_common()) for dom, counts in sorted(per_domain.items())
    }

    return {
        "total_samples": total_samples,
        "language_counts": dict(lang_counts.most_common()),
        "language_percentages": lang_percentages,
        "confidence_stats": confidence_stats,
        "text_length_stats": text_length_stats,
        "per_source_dataset": per_source_sorted,
        "per_domain": per_domain_sorted,
        "num_shards_merged": len(shard_results),
    }


def _add_bar_labels(ax, bars, pcts):
    """Add percentage labels on top of bars."""
    for bar, pct in zip(bars, pcts):
        if pct > 0:
            label = f"{pct:.1f}%" if pct >= 1 else f"{pct:.2f}%"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height(),
                label,
                ha="left",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                rotation=45,
            )


def plot_language_distribution(
    results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Generate language distribution as stacked bars (one bar per language, color = dataset)."""
    # Merge counts across datasets to rank languages
    all_lang_counts: Counter = Counter()
    total_all = 0
    for result in results.values():
        for lang, count in result["language_counts"].items():
            all_lang_counts[lang] += count
        total_all += result["total_samples"]

    top_langs = [lang for lang, _ in all_lang_counts.most_common(TOP_N_LANGUAGES)]
    display_langs = top_langs + ["other"]

    dataset_names = list(results.keys())
    colors = ["steelblue", "darkorange", "seagreen", "firebrick"]

    # Compute per-dataset contribution as % of combined total
    dataset_pcts = {}
    for name, result in results.items():
        pcts = [result["language_counts"].get(lang, 0) / total_all * 100 for lang in top_langs]
        other_count = sum(
            count for lang, count in result["language_counts"].items() if lang not in top_langs
        )
        pcts.append(other_count / total_all * 100)
        dataset_pcts[name] = pcts

    fig, ax = plt.subplots(figsize=(22, 7))
    x = np.arange(len(display_langs))
    width = 0.7

    bottom = np.zeros(len(display_langs))
    for i, name in enumerate(dataset_names):
        pcts = np.array(dataset_pcts[name])
        bars = ax.bar(
            x, pcts, width,
            bottom=bottom,
            label=name,
            color=colors[i % len(colors)],
        )
        # Label each segment with its percentage
        for bar, pct, bot in zip(bars, pcts, bottom):
            if pct > 0:
                label = f"{pct:.1f}%" if pct >= 1 else f"{pct:.2f}%"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + pct,
                    label,
                    ha="left", va="bottom",
                    fontsize=6, fontweight="bold", rotation=45,
                )
        bottom += pcts

    ax.set_yscale("log")
    ax.set_xlabel("Language")
    ax.set_ylabel("Percentage of combined dataset (%, log scale)")
    ax.set_title("Language Distribution in Dolci Datasets (stacked by dataset)")
    ax.set_xticks(x)
    ax.set_xticklabels([lang_label(l) for l in display_langs], rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_ylim(bottom=0.005)

    plt.tight_layout()
    fig.savefig(output_dir / "dolci_language_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved dolci_language_distribution.png")

    # --- Counts plot (same layout, absolute counts instead of percentages) ---
    dataset_counts = {}
    for name, result in results.items():
        counts = [result["language_counts"].get(lang, 0) for lang in top_langs]
        other_count = sum(
            c for lang, c in result["language_counts"].items() if lang not in top_langs
        )
        counts.append(other_count)
        dataset_counts[name] = counts

    fig, ax = plt.subplots(figsize=(22, 7))
    bottom = np.zeros(len(display_langs))
    for i, name in enumerate(dataset_names):
        counts = np.array(dataset_counts[name], dtype=float)
        bars = ax.bar(
            x, counts, width,
            bottom=bottom,
            label=name,
            color=colors[i % len(colors)],
        )
        for bar, cnt, bot in zip(bars, counts, bottom):
            if cnt > 0:
                if cnt >= 1000:
                    label = f"{cnt / 1000:.0f}k" if cnt < 1_000_000 else f"{cnt / 1_000_000:.1f}M"
                else:
                    label = f"{int(cnt)}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + cnt,
                    label,
                    ha="left", va="bottom",
                    fontsize=6, fontweight="bold", rotation=45,
                )
        bottom += counts

    ax.set_yscale("log")
    ax.set_xlabel("Language")
    ax.set_ylabel("Sample count (log scale)")
    ax.set_title("Language Counts in Dolci Datasets (stacked by dataset)")
    ax.set_xticks(x)
    ax.set_xticklabels([lang_label(l) for l in display_langs], rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_ylim(bottom=1)

    plt.tight_layout()
    fig.savefig(output_dir / "dolci_language_counts.png", dpi=150)
    plt.close(fig)
    logger.info("Saved dolci_language_counts.png")


def plot_confidence_distribution(
    results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Generate histogram of detection confidence scores."""
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, results.items()):
        stats = result["confidence_stats"]
        ax.bar(
            ["Mean", "Median", "Low conf %"],
            [stats["mean"], stats["median"], stats["low_confidence_percentage"] / 100],
            color=["steelblue", "darkorange", "firebrick"],
        )
        ax.set_title(f"Confidence Stats - {name}")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score / Fraction")
        for i, v in enumerate(
            [stats["mean"], stats["median"], stats["low_confidence_percentage"] / 100]
        ):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "dolci_confidence_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved dolci_confidence_distribution.png")


def plot_per_group_distribution(
    group_data: dict[str, dict[str, int]],
    group_type: str,
    dataset_name: str,
    output_dir: Path,
) -> None:
    """Generate per-group (domain or source) language distribution plots."""
    for group_name, lang_counts in group_data.items():
        if not lang_counts:
            continue

        total = sum(lang_counts.values())
        # Show top languages for this group
        top_items = sorted(lang_counts.items(), key=lambda x: -x[1])[:TOP_N_LANGUAGES]
        langs = [item[0] for item in top_items]
        pcts = [item[1] / total * 100 for item in top_items]

        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(langs, pcts, color="steelblue")
        _add_bar_labels(ax, bars, pcts)

        safe_name = group_name.replace("/", "_").replace(" ", "_")
        ax.set_title(f"Language Distribution - {dataset_name} - {group_type}: {group_name}")
        ax.set_xlabel("Language")
        ax.set_ylabel("Percentage (%, log scale)")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.005)
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels([lang_label(l) for l in langs], rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3, which="both")
        plt.tight_layout()
        fig.savefig(
            output_dir / f"dolci_lang_dist_{group_type}_{safe_name}.png",
            dpi=150,
        )
        plt.close(fig)

    logger.info("Saved %d %s plots for %s", len(group_data), group_type, dataset_name)


def process_dataset(
    dataset_key: str,
    output_dir: Path,
    num_workers: int,
    sample: int,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict:
    """Process a single dataset (or shard): load, detect languages, aggregate, save."""
    dataset_name = DATASET_NAMES[dataset_key]
    logger.info("Loading dataset: %s", dataset_name)
    ds = load_dataset(dataset_name, split="train")
    logger.info("Loaded %d samples total", len(ds))

    if sample > 0:
        ds = ds.select(range(min(sample, len(ds))))
        logger.info("Sampled %d rows", len(ds))

    # Shard the dataset
    if num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_index, contiguous=True)
        logger.info("Shard %d/%d: %d samples", shard_index, num_shards, len(ds))

    logger.info("Extracting user text...")
    ds = ds.map(extract_user_text, num_proc=num_workers)

    logger.info("Detecting languages...")
    ds = ds.map(
        detect_languages_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_workers,
    )

    logger.info("Aggregating results...")
    results = aggregate_results(ds)
    results["dataset"] = dataset_name
    results["dataset_key"] = dataset_key
    results["shard_index"] = shard_index
    results["num_shards"] = num_shards

    # Save JSON — use shard suffix when sharding
    if num_shards > 1:
        output_file = output_dir / f"dolci_{dataset_key}_lang_dist_shard{shard_index}.json"
    else:
        output_file = output_dir / f"dolci_{dataset_key}_lang_dist.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved results to %s", output_file)

    return results


def cmd_detect(args: argparse.Namespace) -> None:
    """Run language detection on a dataset (or shard)."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_process = (
        ["instruct", "think"] if args.dataset == "both" else [args.dataset]
    )

    all_results = {}
    for ds_key in datasets_to_process:
        result = process_dataset(
            ds_key,
            args.output_dir,
            args.num_workers,
            args.sample,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        all_results[ds_key] = result

    # Only generate plots when not sharding (single-node mode)
    if args.num_shards <= 1:
        generate_plots(all_results, args.output_dir)

    print_summary(all_results)
    logger.info("Done! Results saved to %s", args.output_dir)


def cmd_merge(args: argparse.Namespace) -> None:
    """Merge shard results and generate plots."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_process = (
        ["instruct", "think"] if args.dataset == "both" else [args.dataset]
    )

    all_results = {}
    for ds_key in datasets_to_process:
        # Find all shard files for this dataset
        shard_files = sorted(args.output_dir.glob(f"dolci_{ds_key}_lang_dist_shard*.json"))
        if not shard_files:
            # Try non-sharded file (single-node run)
            single = args.output_dir / f"dolci_{ds_key}_lang_dist.json"
            if single.exists():
                with open(single) as f:
                    all_results[ds_key] = json.load(f)
                logger.info("Loaded single-node results for %s", ds_key)
                continue
            logger.warning("No shard files found for %s", ds_key)
            continue

        logger.info("Found %d shards for %s", len(shard_files), ds_key)
        shard_results = []
        for sf in shard_files:
            with open(sf) as f:
                shard_results.append(json.load(f))

        merged = merge_shard_results(shard_results)
        merged["dataset"] = DATASET_NAMES[ds_key]
        merged["dataset_key"] = ds_key

        # Save merged result
        output_file = args.output_dir / f"dolci_{ds_key}_lang_dist.json"
        with open(output_file, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        logger.info("Saved merged results to %s", output_file)

        all_results[ds_key] = merged

    if all_results:
        generate_plots(all_results, args.output_dir)
        print_summary(all_results)

    logger.info("Merge complete! Results saved to %s", args.output_dir)


def generate_plots(all_results: dict[str, dict], output_dir: Path) -> None:
    """Generate all plots from aggregated results."""
    logger.info("Generating plots...")
    plot_language_distribution(all_results, output_dir)
    plot_confidence_distribution(all_results, output_dir)

    for ds_key, result in all_results.items():
        if result["per_domain"]:
            plot_per_group_distribution(
                result["per_domain"], "domain", ds_key, output_dir
            )


def print_summary(all_results: dict[str, dict]) -> None:
    """Print a human-readable summary of results."""
    for ds_key, result in all_results.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {result['dataset']} ({result['total_samples']} samples)")
        print(f"{'='*60}")
        print("Top languages:")
        for lang, pct in list(result["language_percentages"].items())[:10]:
            count = result["language_counts"][lang]
            print(f"  {lang:>10}: {pct:>7.3f}% ({count:>10,} samples)")
        stats = result["confidence_stats"]
        print(f"\nConfidence: mean={stats['mean']:.3f}, median={stats['median']:.3f}")
        print(
            f"Low confidence (<0.5): {stats['low_confidence_count']:,} "
            f"({stats['low_confidence_percentage']:.2f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect language distribution in Dolci datasets")
    subparsers = parser.add_subparsers(dest="command")

    # --- detect subcommand ---
    detect_parser = subparsers.add_parser("detect", help="Run language detection")
    detect_parser.add_argument(
        "--dataset",
        choices=["instruct", "think", "both"],
        default="both",
        help="Which dataset to analyze",
    )
    detect_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("oellm/analysis/dolci_distribution/results"),
    )
    detect_parser.add_argument("--num-workers", type=int, default=32)
    detect_parser.add_argument(
        "--sample", type=int, default=0, help="0 = full dataset"
    )
    detect_parser.add_argument(
        "--shard-index", type=int, default=0, help="Shard index (0-based)"
    )
    detect_parser.add_argument(
        "--num-shards", type=int, default=1, help="Total number of shards"
    )

    # --- merge subcommand ---
    merge_parser = subparsers.add_parser("merge", help="Merge shard results and plot")
    merge_parser.add_argument(
        "--dataset",
        choices=["instruct", "think", "both"],
        default="both",
    )
    merge_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("oellm/analysis/dolci_distribution/results"),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Default to detect if no subcommand given (backwards compat)
    if args.command is None or args.command == "detect":
        if args.command is None:
            # Set defaults for legacy invocation
            args.output_dir = Path("oellm/analysis/dolci_distribution/results")
            args.dataset = "both"
            args.num_workers = 32
            args.sample = 0
            args.shard_index = 0
            args.num_shards = 1
        cmd_detect(args)
    elif args.command == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
