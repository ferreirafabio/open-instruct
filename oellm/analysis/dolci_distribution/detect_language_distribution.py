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
TOP_N_LANGUAGES = 15


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
                results = detect(text)
                # detect() returns a list of dicts: [{"lang": "en", "score": 0.99}]
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


def plot_language_distribution(
    results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Generate overall language distribution bar chart comparing datasets."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Collect all languages across datasets, sorted by total count
    all_lang_counts: Counter = Counter()
    for name, result in results.items():
        for lang, count in result["language_counts"].items():
            all_lang_counts[lang] += count

    top_langs = [lang for lang, _ in all_lang_counts.most_common(TOP_N_LANGUAGES)]

    x = np.arange(len(top_langs))
    width = 0.35
    dataset_names = list(results.keys())

    for i, name in enumerate(dataset_names):
        pcts = [results[name]["language_percentages"].get(lang, 0) for lang in top_langs]
        offset = (i - (len(dataset_names) - 1) / 2) * width
        bars = ax.bar(x + offset, pcts, width, label=name)
        for bar, pct in zip(bars, pcts):
            if pct > 1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Language")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Language Distribution in Dolci Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(top_langs, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "dolci_language_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved dolci_language_distribution.png")


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

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(langs, pcts, color="steelblue")
        for i, (lang, pct) in enumerate(zip(langs, pcts)):
            if pct > 1:
                ax.text(i, pct, f"{pct:.1f}%", ha="center", va="bottom", fontsize=7)

        safe_name = group_name.replace("/", "_").replace(" ", "_")
        ax.set_title(f"Language Distribution - {dataset_name} - {group_type}: {group_name}")
        ax.set_xlabel("Language")
        ax.set_ylabel("Percentage (%)")
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels(langs, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
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
