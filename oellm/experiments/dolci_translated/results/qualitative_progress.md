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
