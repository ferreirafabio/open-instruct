"""Generate qualitative completions across 12 models × ~274 LMArena prompts (fr/de/fi).

Loads each model with vLLM, runs all prompts, writes results incrementally to
completions.json so a requeue resumes from where it stopped.

Models without a chat template (OLMo-3 base) get the raw prompt; others get
apply_chat_template(messages, add_generation_prompt=True).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path("/work/dlclarge2/ferreira-oellm/open-instruct")
DEFAULT_PROMPTS = PROJECT_ROOT / "oellm/experiments/dolci_translated/site/prompts_lmarena.json"
DEFAULT_OUT = PROJECT_ROOT / "oellm/experiments/dolci_translated/site/completions.json"

CKPT_BASE = PROJECT_ROOT / "checkpoints/ferreira/olmo3-7b-sft"
OLMO3_BASE_PATH = (
    PROJECT_ROOT
    / "models/baselines/Olmo-3-1025-7B/models--allenai--Olmo-3-1025-7B/snapshots/18b40a1e895f829c68a132befa20109c41488e62"
)
SFT_BASELINE_PATH = PROJECT_ROOT / "models/eval/instruct-v2-step3252"


@dataclass
class ModelSpec:
    id: str
    label: str
    group: str  # "base", "sft-baseline", "A-75en", "A-25en"
    step: int | None
    path: Path

    def to_meta(self) -> dict:
        return {"id": self.id, "label": self.label, "group": self.group, "step": self.step}


def build_model_list() -> list[ModelSpec]:
    models: list[ModelSpec] = []

    models.append(
        ModelSpec(
            id="olmo3-base",
            label="OLMo-3-1025-7B (pre-SFT base)",
            group="base",
            step=None,
            path=OLMO3_BASE_PATH,
        )
    )
    models.append(
        ModelSpec(
            id="sft-baseline-step3252",
            label="OLMo-3-7B-Instruct-SFT-v2 (step 3252)",
            group="sft-baseline",
            step=3252,
        path=SFT_BASELINE_PATH,
        )
    )

    a75_steps = [500, 1500, 2500, 3500, 3998]
    for step in a75_steps:
        if step == 3998:
            path = CKPT_BASE / "dolci-translated-A-75en-hf"
        else:
            path = CKPT_BASE / f"dolci-translated-A-75en-step{step}-hf"
        models.append(
            ModelSpec(
                id=f"A-75en-step{step}",
                label=f"A-75en step {step}",
                group="A-75en",
                step=step,
                path=path,
            )
        )

    a25_steps = [500, 2500, 5000, 7000, 8686]
    for step in a25_steps:
        if step == 8686:
            path = CKPT_BASE / "dolci-translated-A-25en-hf"
        else:
            path = CKPT_BASE / f"dolci-translated-A-25en-step{step}-hf"
        models.append(
            ModelSpec(
                id=f"A-25en-step{step}",
                label=f"A-25en step {step}",
                group="A-25en",
                step=step,
                path=path,
            )
        )

    return models


def load_state(out_path: Path) -> dict:
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)
    return {"models": [], "prompts": [], "completions": {}}


def save_state(state: dict, out_path: Path) -> None:
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=None, help="Cap prompts per language for dry runs")
    parser.add_argument("--only-model", default=None, help="Run a single model id (skip others)")
    parser.add_argument("--skip-group", action="append", default=[], help="Skip an entire group (e.g. A-25en). Repeatable.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(prompts_path, encoding="utf-8") as f:
        prompts_payload = json.load(f)
    all_prompts: list[dict] = prompts_payload["prompts"]

    if args.limit is not None:
        per_lang: dict[str, list[dict]] = {}
        for p in all_prompts:
            per_lang.setdefault(p["lang"], []).append(p)
        capped: list[dict] = []
        for lang, items in per_lang.items():
            capped.extend(items[: args.limit])
        all_prompts = capped
        print(f"[dry-run] capped to {len(all_prompts)} prompts ({args.limit}/lang)")

    models = build_model_list()
    if args.skip_group:
        skip = set(args.skip_group)
        models = [m for m in models if m.group not in skip]
        print(f"Skipping groups: {', '.join(sorted(skip))} → {len(models)} models remain")
    if args.only_model:
        models = [m for m in models if m.id == args.only_model]
        if not models:
            sys.exit(f"--only-model {args.only_model} not found")

    state = load_state(out_path)
    state["models"] = [m.to_meta() for m in models]
    state["prompts"] = all_prompts if not args.limit else state.get("prompts", []) or all_prompts
    completions: dict[str, str] = state.setdefault("completions", {})

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    for m_idx, model in enumerate(models):
        if not model.path.exists():
            print(f"SKIP: {model.id} -- path missing: {model.path}")
            continue

        # Find which prompts still need generation for this model
        todo: list[dict] = [p for p in all_prompts if f"{model.id}::{p['lang']}::{p['idx']}" not in completions]
        if not todo:
            print(f"[{m_idx + 1}/{len(models)}] {model.id}: all {len(all_prompts)} prompts done, skipping")
            continue

        print(f"\n[{m_idx + 1}/{len(models)}] Loading {model.id} from {model.path}")
        print(f"   {len(todo)}/{len(all_prompts)} prompts pending")

        llm = LLM(
            model=str(model.path),
            gpu_memory_utilization=0.85,
            max_model_len=8192,
            enforce_eager=True,
        )
        tokenizer = llm.get_tokenizer()
        has_chat = getattr(tokenizer, "chat_template", None) is not None

        # Format prompts + filter out anything that wouldn't fit (model max is 8192)
        max_input_tokens = 8192 - args.max_tokens - 16  # leave a small margin
        formatted: list[str] = []
        formatted_prompts: list[dict] = []
        skipped: list[tuple[dict, int]] = []
        for p in todo:
            if has_chat:
                msgs = [{"role": "user", "content": p["prompt"]}]
                txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            else:
                txt = p["prompt"]
            n_tok = len(tokenizer.encode(txt, add_special_tokens=False))
            if n_tok > max_input_tokens:
                skipped.append((p, n_tok))
                continue
            formatted.append(txt)
            formatted_prompts.append(p)
        if skipped:
            print(f"   skipping {len(skipped)} oversized prompts (>{max_input_tokens} tokens)")
            for p, n_tok in skipped[:5]:
                print(f"     - {p['lang']} #{p['idx']}: {n_tok} tokens")

        outputs = llm.generate(formatted, sampling_params)
        for p, out in zip(formatted_prompts, outputs):
            key = f"{model.id}::{p['lang']}::{p['idx']}"
            completions[key] = out.outputs[0].text.strip()

        save_state(state, out_path)
        print(f"   saved {len(todo)} new completions (total: {len(completions)})")

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nAll done. {len(completions)} total completions in {out_path}")


if __name__ == "__main__":
    main()
