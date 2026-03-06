# Multilingual EU — Experiment Progress

**Started:** 2026-03-05
**Owner:** ferreira
**Cluster:** kislurm (alldlc2_gpu-h200 / alldlc2_cpu-epyc9655)

---

## Evaluation Benchmark

**m-arena-hard-EU** covers **12 languages**: cs, de, el, en, es, fr, it, nl, pl, pt, ro, uk

| Role | Languages | Count |
|------|-----------|-------|
| Training languages | cs, de, es, fr, it, nl, pl, pt | 8 |
| Held-out (zero-shot) | el (Greek), ro (Romanian) | 2 |
| Benchmark-only | en (English baseline), uk (Ukrainian) | 2 |

---

## Pipeline Status

| Step | Script | Status | SLURM Job | Duration | Notes |
|------|--------|--------|-----------|----------|-------|
| 1a. Profile datasets | `profile_multilingual_slurm.sh` | Done | 27476899 (array 0-3) | 1-4 min | 4 CPU nodes parallel |
| 1b. Merge profiles | `merge_profiles.sh` | Done | 27476900 | 3 sec | Merged into `dataset_profiles.json` |
| 2. Preprocess | `preprocess_multilingual_slurm.sh` | Done | 27477043 (array 0-3) | 10s-2min | 4 CPU nodes parallel |
| 3. Split by language | `split_multilingual_slurm.sh` | Done | 27477209 (array 0-3) | 3-26 sec | 4 CPU nodes parallel |
| 4. Assemble A1-90en | `assemble_mixture.py` | Done (v2) | (local) | ~30 sec | 94,694 samples, equal EU distribution |
| 5. Tokenize A1-90en | `tokenize_A1-90en.sh` | Done (v2) | 27477424 | ~7 min | Re-tokenized with equal EU ratios |
| 6. Train A1-90en | `train_multilingual_sft_slurm.sh` | Done | 27477567 | ~64 min | 238 steps, final loss=0.78, PPL=2.19 |
| 7. Convert + Eval A1-90en | `convert_and_eval_A1.sh` | **Running** | 27479949 | — | Convert to HF, then m-arena-hard-EU winrate |

---

## Experiment Matrix

All experiments use **equal EU language distribution** (EU% / 8 per language).

### Track A: fusion-synth primary (~94.7k samples)

| Exp | En/EU | Per-EU-lang % | Config | Data | Train | Eval | Notes |
|-----|-------|--------------|--------|------|-------|------|-------|
| **A1** | **90/10** | **1.25%** | `multilingual_trackA_90en.yaml` | **Done** (v2) | Pending | Pending | Baseline multilingual signal |
| A2 | 80/20 | 2.5% | `multilingual_trackA_80en.yaml` | Pending | Pending | Pending | More EU data |
| A3 | 70/30 | 3.75% | `multilingual_trackA_70en.yaml` | Pending | Pending | Pending | Aggressive EU |

### Track B: All 4 sources (~500k samples) — starts after Track A results

| Exp | En/EU | Per-EU-lang % | Config | Data | Train | Eval | Notes |
|-----|-------|--------------|--------|------|-------|------|-------|
| B1 | 90/10 | 1.25% | `multilingual_trackB_90en.yaml` | Pending | Pending | Pending | Scale with more sources |
| B2 | 80/20 | 2.5% | `multilingual_trackB_80en.yaml` | Pending | Pending | Pending | Scale + more EU |

---

## A1-90en Details

### Data Mixture (v2 — equal EU distribution)

| Language | Target % | Samples | Source(s) |
|----------|----------|---------|-----------|
| en | 90.0% | 85,230 | fusion-synth + wildchat + lmsys |
| de | 1.25% | 1,183 | fusion-synth |
| fr | 1.25% | 1,183 | fusion-synth |
| es | 1.25% | 1,183 | fusion-synth |
| it | 1.25% | 1,183 | fusion-synth |
| pt | 1.25% | 1,183 | fusion-synth |
| pl | 1.25% | 1,183 | wildchat + lmsys |
| nl | 1.25% | 1,183 | wildchat + lmsys |
| cs | 1.25% | 1,183 | wildchat + lmsys |
| **Total** | **100%** | **94,694** | |

### Training Config
- Continued SFT from `dolci-instruct-sft-v2-horeka`
- Learning rate: 8e-5
- Epochs: 2
- Sequence length: 32,768
- Batch size: 1M tokens (seq_len x 32)
- GPUs: 8x H200
- Checkpoint interval: every 50 steps
- Estimated training time: ~1.7 hours

### Evaluation Plan
- **m-arena-hard-EU** (12 languages): winrate + rubric
- **arena-hard** (English only): regression check
- Judge: `Qwen/Qwen3.5-35B-A3B`
- Evaluate intermediate checkpoints (every 50 steps) for training curves

---

## Data Shortfall (cs, nl)

Czech and Dutch have limited data across all 4 sources (cs: 1,295 total, nl: 2,800 total). At higher EU ratios, targets cannot be met — missing samples are dropped, shrinking the total dataset. No upsampling is applied.

| Exp | Target/lang | cs actual | nl actual | Total actual | vs target |
|-----|------------|-----------|-----------|-------------|-----------|
| A1 (10% EU) | 1,183 | 1,183 (100%) | 1,183 (100%) | 94,694 | -6 |
| A2 (20% EU) | 2,367 | 1,295 (55%) | 2,367 (100%) | 93,624 | -1,072 |
| A3 (30% EU) | 3,551 | 1,295 (36%) | 2,800 (79%) | 91,691 | -3,007 |

**Impact on actual per-language distribution:**

A2-80en: de/fr/es/it/pt/pl/nl = 2.53% each, **cs = 1.38%**, total EU = 19.1% (not 20%)

A3-70en: de/fr/es/it/pt/pl = 3.87% each, **nl = 3.05%**, **cs = 1.41%**, total EU = 27.7% (not 30%)

**Decision:** Accepted as-is. cs and nl are inherently data-scarce. Results for these languages should be interpreted with this caveat. Cross-experiment comparisons for cs gains (A1→A2→A3) are limited since its actual ratio barely changes.

---

## Next Steps

1. Wait for A1-90en tokenization (27477424), then submit training
2. After training: convert checkpoints to HF, evaluate
3. Assemble + tokenize A2-80en and A3-70en (can be done in parallel now)
4. Based on A1-A3 results, decide whether to proceed to Track B

---

## Log

| Date | Time | Event |
|------|------|-------|
| 2026-03-05 | 00:57 | Profiling jobs submitted (27476899) |
| 2026-03-05 | 01:01 | Profiling complete, profiles merged (27476900) |
| 2026-03-05 | 01:06 | Preprocessing jobs submitted (27477043) |
| 2026-03-05 | 01:08 | Preprocessing complete (all 4 datasets) |
| 2026-03-05 | 01:10 | Split-by-language v1 submitted (27477148) — wrote single row groups |
| 2026-03-05 | 01:15 | Fixed pyarrow nested chunked array bug: split now writes row_group_size=10000 |
| 2026-03-05 | 01:16 | Split-by-language v2 submitted (27477209) — fixed row groups |
| 2026-03-05 | 01:17 | Split complete (all 4 datasets) |
| 2026-03-05 | 01:24 | A1-90en mixture assembled v1 (94,698 samples, unequal EU ratios) |
| 2026-03-05 | 01:25 | A1-90en tokenization v1 started (27477255) |
| 2026-03-05 | 01:32 | A1-90en tokenization v1 complete (125M tokens) |
| 2026-03-05 | 01:33 | A1-90en training v1 submitted (27477356, 8xH200) — unequal EU ratios |
| 2026-03-05 | 01:40 | **Changed to equal EU distribution** (1.25% each for A1). Cancelled 27477356 |
| 2026-03-05 | 01:40 | Updated all 5 YAML configs (A1-A3, B1-B2) to equal EU distribution |
| 2026-03-05 | 01:41 | A1-90en mixture re-assembled v2 (94,694 samples, 1,183/lang) |
| 2026-03-05 | 01:41 | A1-90en tokenization v2 submitted (27477424) |
| 2026-03-05 | 01:48 | A1-90en training v2 submitted (27477520) — failed: omegaconf 2.0.6 missing `to_object()` |
| 2026-03-05 | 01:50 | Upgraded omegaconf 2.0.6 → 2.3.0 (`uv pip install "omegaconf>=2.2"`) |
| 2026-03-05 | 01:51 | A1-90en training v3 submitted (27477567, 8xH200) — running |
| 2026-03-05 | 03:01 | A1-90en training complete (238 steps, ~64 min, final loss=0.78) |
| 2026-03-05 | 03:02 | A1-90en convert + eval submitted (27479949) — converts step200+238, evals step238 |
| 2026-03-05 | 03:05 | Convert failed: `max_position_embeddings` null in HF config. Fixed to 65536 |
| 2026-03-05 | 03:06 | A1-90en eval resubmitted (27490453) — winrate mode |
| 2026-03-05 | 03:07 | A2-80en + A3-70en assembled. cs short: 1,295/2,367 (A2), 1,295/3,551 (A3). nl short in A3: 2,800/3,551 |
| 2026-03-05 | 03:08 | A2+A3 tokenization submitted (27490894, array 0-1) |

---

## Known Issues / Fixes

1. **pyarrow nested chunked array bug**: `pq.read_table` fails on large parquets with nested types (list of structs) written as a single row group. Fix: write with `row_group_size=10000` in split step, use `iter_batches` in assembly.
2. **Language name mapping**: WildChat and lmsys-chat use full language names ("English", "German"), oasst2 uses variants ("pt-BR", "uk-UA"). Fixed with `LANGUAGE_NAME_TO_ISO` mapping dict and `.split("-")[0]` normalization.
3. **trust_remote_code deprecation**: Some HF datasets raise `ValueError: trust_remote_code is not supported anymore`. Fixed with try/except fallback.
4. **Schema column order mismatch**: Different datasets write columns in different order (messages,language vs language,messages). Fixed with schema unification before `pa.concat_tables`.
5. **omegaconf version mismatch**: OLMo-core calls `OmegaConf.to_object()` (requires >=2.2), but env had 2.0.6. Fix: `uv pip install "omegaconf>=2.2"` → upgraded to 2.3.0.
