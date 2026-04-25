# Qualitative Completions Webpage — Progress

## Goal
Filterable static webpage at `ferreirafabio.github.io/oellm-completions/` showing
side-by-side completions across 12 models for fr/de/fi LMArena prompts.

## Pipeline Status

| Step | Script | Status | Job | Notes |
|------|--------|--------|-----|-------|
| 1. Sample LMArena prompts | `scripts/qualitative/sample_lmarena_prompts.py` | Done | (local) | 274 prompts (100 fr + 100 de + 74 fi). Wrote `site/prompts_lmarena.json`. Finnish capped at 74 after dedup (LMArena has 96 fi entries total). |
| 2. Convert intermediate ckpts | `scripts/qualitative/convert_intermediate_ckpts.sh` | Running | 28802199 (array 0-7) | 8 conversions on `alldlc2_cpu-epyc9655`. ~1h each. |
| 3. Generate completions | `scripts/qualitative/generate_completions.sh` | Pending | — | 12 models × 274 prompts ≈ 3,288 generations on `alldlc2_gpu-h200`. Submit after step 2 finishes. |
| 4. Build webpage bundle | `site/index.html` + `app.js` + `style.css` | Done (UI scaffold) | — | Renders against placeholder `completions.json`. Will pick up real completions once generation finishes. |
| 5. Deploy | `site/README.md` | Pending | — | User pushes to `ferreirafabio.github.io` repo (not cloned here). |

## Models (12 total)

| Group | Model | Step | HF path |
|-------|-------|------|---------|
| base | OLMo-3-1025-7B | n/a | `models/baselines/Olmo-3-1025-7B/.../snapshots/18b40a1e.../` |
| sft-baseline | OLMo-3-7B-Instruct-SFT-v2 | 3252 | `models/eval/instruct-v2-step3252` |
| A-75en | dolci_translated A-75en | 500, 1500, 2500, 3500, 3998 | `checkpoints/.../dolci-translated-A-75en{-step{N}-hf,-hf}` |
| A-25en | dolci_translated A-25en | 500, 2500, 5000, 7000, 8686 | `checkpoints/.../dolci-translated-A-25en{-step{N}-hf,-hf}` |

## Prompts

| Lang | Count | Source |
|------|-------|--------|
| fr | 100 | lmarena-ai/arena-human-preference-100k (filter language=French) |
| de | 100 | lmarena-ai/arena-human-preference-100k (filter language=German) |
| fi | 74 | lmarena-ai/arena-human-preference-100k (filter language=Finnish, all unique user prompts) |

Sampling seed: 42. First user-turn extracted, deduplicated.

## Generation Settings

- vLLM, temperature 0.0, max_tokens 1024, max_model_len 4096
- Chat template applied for instruct models; raw prompt for OLMo-3 base
- Single H200 GPU, ~6h estimated for the full run
- Incremental writes to `site/completions.json` (resumable across requeues)

## Final state (2026-04-23 23:13 UTC)

- **3288 / 3288 completions** (12 models × 274 prompts), all per-model counts equal to 274
- Jobs: conversion array `28802199` (8 tasks, ~10 min total on `alldlc2_cpu-epyc9655`); generation `28802214` (~16 min on a single H200)
- Site is ready to deploy via `site/README.md`

## Major refactor (2026-04-24)

- Renamed deployment paths: `oellm-completions` → `olmo3-multilingual-dolci-sft-progression` on both github.io and HF Space (`move_repo` for HF, redirect stub for github.io)
- Stratified-by-category prompt re-sampling: 3 langs (274 prompts) → **7 langs (698 prompts)** with target ≥50 prompts per LMArena category per language
- Added Swedish, Italian, Spanish, Czech as new languages
- Added LMArena category filter (multi-select pills)
- Original A-25en card hidden pending matched-compute re-run

## Matched A-25en submitted (2026-04-24)

- Original A-25en used 4.62M samples (8686 steps) vs A-75en's 2.87M (3998 steps) → 1.6× more compute
- Submitted matched re-run: same 2.87M total samples, 25/75 ratio (717k English sampled + 2.15M translated sampled), `PERMANENT_SAVE_INTERVAL=500` to match A-75en's checkpointing pattern
- Pipeline chain: assemble (`28807731`) → tokenize (`28807733`) → SFT training (`28807746`)

## Matched A-25en complete (2026-04-25)

- Training started 2026-04-24 19:32, ended 2026-04-25 ~16:00, **final step = 5398**
- A-75en converged in **3998 steps**, A-25en-matched in **5398 steps** despite both runs processing the same 2.87M samples × 2 epochs at the same 1M-token batch size and identical hyperparameters (lr=8e-5, AdamW, linear warmup, bf16, etc.). The asymmetry comes from translated text packing into more tokens per sample → more sequences per epoch → more steps to cover the same data.
- Intermediate ckpts converted + generated incrementally during training: step500 (parallel chain `28811481`/`28811482`), then step1500/2500/3500 (jobs `28812804`/`805`/`806`/`807`), final step5398 post-training (jobs `28815065` convert + `28816548` gen).
- Site updated to use matched A-25en across all 7 langs.

### Comparison philosophy
The viewer's slider compares **literal step numbers** at ticks 0-3 (both groups at step 500 / 1500 / 2500 / 3500). At those ticks A-75en is at 12.5% / 37.5% / 62.5% / 87.5% trained, while A-25en-matched is at 9.3% / 27.8% / 46.3% / 64.8%. The two runs are NOT at identical training fractions at intermediate ticks.

**Tick 4 (final)** is the only slider position where both runs are at 100% trained — that's the cleanest apples-to-apples comparison. The matched-compute property (same data, same epochs) means the two runs are equivalent at their respective ends, even though they took different numbers of steps to get there.

If a per-tick training-fraction match is needed in the future, re-pick A-25en-matched intermediates at A-75en's fractions (e.g. step 5398 × 0.125 ≈ 675 instead of 500, step 5398 × 0.375 ≈ 2024 instead of 1500). Costs 4 more conversions + ~5 min H200 of regen.

### Hero banner
Added a visible note to the viewer hero explaining the step-count asymmetry and the tick-4-is-true-final caveat, with a Playwright test asserting the explanation stays present.
