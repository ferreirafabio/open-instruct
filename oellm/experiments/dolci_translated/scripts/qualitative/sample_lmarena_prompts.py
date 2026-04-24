"""Stratified-by-category LMArena prompt sampler for the qualitative viewer.

Output: a single JSON containing all prompts across 7 EU languages, each tagged
with its LMArena category flags. Each language is sampled with a target of
PER_CATEGORY_TARGET prompts hitting each major category; languages with
insufficient pool size fall back to "take everything available".

Source: lmarena-ai/arena-human-preference-100k
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

LANG_NAME = {
    "fr": "French",
    "de": "German",
    "fi": "Finnish",
    "sv": "Swedish",
    "it": "Italian",
    "es": "Spanish",
    "cs": "Czech",
}

# Major LMArena category tags we care about for the viewer filter.
# Each prompt may have any number of these set to True.
CATEGORIES = [
    ("math", "math_v0.1", "math"),
    ("if", "if_v0.1", "if"),
    ("creativity", "criteria_v0.1", "creativity"),
    ("complexity", "criteria_v0.1", "complexity"),
    ("problem_solving", "criteria_v0.1", "problem_solving"),
    ("technical_accuracy", "criteria_v0.1", "technical_accuracy"),
    ("specificity", "criteria_v0.1", "specificity"),
    ("real_world", "criteria_v0.1", "real_world"),
    ("domain_knowledge", "criteria_v0.1", "domain_knowledge"),
]

PER_CATEGORY_TARGET = 50
SEED = 42


def load_lmarena() -> pd.DataFrame:
    path = snapshot_download(
        repo_id="lmarena-ai/arena-human-preference-100k",
        repo_type="dataset",
        allow_patterns="*parquet",
        force_download=False,
    )
    return pd.read_parquet(Path(path) / "data" / "arena-explorer-preference-100k.parquet")


def first_user_turn(conv) -> str | None:
    if conv is None or len(conv) == 0:
        return None
    for turn in conv:
        if turn.get("role") == "user":
            text = (turn.get("content") or "").strip()
            return text or None
    return None


def extract_categories(category_tag) -> list[str]:
    if not isinstance(category_tag, dict):
        return []
    out: list[str] = []
    for label, tagger_key, sub_key in CATEGORIES:
        tagger = category_tag.get(tagger_key)
        if isinstance(tagger, dict) and tagger.get(sub_key) is True:
            out.append(label)
    return out


def sample_lang(df: pd.DataFrame, lang_code: str, rng: random.Random) -> list[dict]:
    """Stratified sample: include every unique-prompt row in this language; for
    each major category, ensure up to PER_CATEGORY_TARGET prompts hitting it
    are selected (selecting greedily, multi-label aware)."""

    full_name = LANG_NAME[lang_code]
    sub = df[df["language"] == full_name]

    # Build candidate pool: dedupe by first-user-turn text
    pool: list[dict] = []
    seen: set[str] = set()
    for _, row in sub.iterrows():
        prompt = first_user_turn(row["conversation_a"])
        if prompt is None or prompt in seen:
            continue
        seen.add(prompt)
        cats = extract_categories(row["category_tag"])
        pool.append({
            "qid": str(row["question_id"]),
            "prompt": prompt,
            "categories": cats,
        })

    rng.shuffle(pool)

    # Greedy stratified pick.
    picked_idx: set[int] = set()
    cat_counts: dict[str, int] = {label: 0 for label, _, _ in CATEGORIES}

    # 1) For each category, fill up to target
    for label, _, _ in CATEGORIES:
        for i, p in enumerate(pool):
            if cat_counts[label] >= PER_CATEGORY_TARGET:
                break
            if label in p["categories"] and i not in picked_idx:
                picked_idx.add(i)
                for c in p["categories"]:
                    cat_counts[c] = cat_counts.get(c, 0) + 1

    # 2) For low-resource languages where the per-category target couldn't be
    #    met (e.g. Swedish, Finnish), fill out with whatever else we have so
    #    the viewer is still useful.
    if len(picked_idx) < min(100, len(pool)):
        for i, _ in enumerate(pool):
            if len(picked_idx) >= min(100, len(pool)):
                break
            if i not in picked_idx:
                picked_idx.add(i)

    selected = [pool[i] for i in sorted(picked_idx)]

    return [
        {
            "idx": i,
            "lang": lang_code,
            "prompt": p["prompt"],
            "source_question_id": p["qid"],
            "categories": p["categories"],
        }
        for i, p in enumerate(selected)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent.parent.parent / "site" / "prompts_lmarena.json"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading LMArena dataset...")
    df = load_lmarena()
    print(f"  {len(df)} rows total")

    rng = random.Random(SEED)
    all_prompts: list[dict] = []
    summary: list[tuple[str, int, dict[str, int]]] = []

    for lang_code in LANG_NAME.keys():
        picked = sample_lang(df, lang_code, rng)
        # Re-index globally (one global index space, then per-lang in viewer)
        for p in picked:
            p["idx"] = len(all_prompts) + p["idx"]
        # Re-index within language so the viewer's #1..#N counter aligns per lang
        for new_i, p in enumerate(picked):
            p["idx"] = new_i  # per-language idx (matches existing viewer expectation)

        cat_counts = {label: 0 for label, _, _ in CATEGORIES}
        for p in picked:
            for c in p["categories"]:
                cat_counts[c] = cat_counts.get(c, 0) + 1
        summary.append((lang_code, len(picked), cat_counts))
        all_prompts.extend(picked)

    print()
    print("Per-language counts:")
    for lang_code, n, cats in summary:
        print(f"  {lang_code} ({LANG_NAME[lang_code]}): {n} prompts")
        for label, _, _ in CATEGORIES:
            print(f"    - {label}: {cats.get(label, 0)}")

    payload = {
        "source": "lmarena-ai/arena-human-preference-100k",
        "seed": SEED,
        "per_category_target": PER_CATEGORY_TARGET,
        "categories": [label for label, _, _ in CATEGORIES],
        "prompts": all_prompts,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(all_prompts)} prompts to {out_path}")


if __name__ == "__main__":
    main()
