## Overview

We're investigating how to add multilingual capabilities to OLMo-3-7B without catastrophic forgetting of English. Starting from our reproduced instruct SFT checkpoint, we continue training on mixtures of English + EU-language data at varying ratios.

### Background: Dolci-Instruct-SFT is 93% English

The base checkpoint's training data (2.15M samples) is overwhelmingly English with a long tail of ~160 languages, each below ~0.4% (Spanish is 2nd largest with ~0.39% (instruct) ~0.41% (think)).

Count-based:
![Dolci-Instruct-SFT language counts](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/dolci_distribution/results/dolci_language_counts.png?raw=true)

Percentage-based (w.r.t. the respective dataset):
![Dolci-Instruct-SFT language percentage](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/dolci_distribution/results/dolci_language_distribution.png?raw=true)

### Setup

| | |
|---|---|
| **Base checkpoint** | OLMo-3-7B-Instruct-SFT (reproduced, [details](https://github.com/allenai/open-instruct/issues/1352#issuecomment-2823953936)) |
| **Judge** | Qwen3-30B-A3B-Instruct-2507 (VLLM, winrate mode, both ordering, 8k max tokens) |
| **Benchmarks** | m-arena-hard-EU (6000 prompts, 12 EU languages), arena-hard (500 prompts, English) |
| **Training languages** | en, de, es, fr, it, pt, pl, nl, cs  (ratio varies by experiment) |
| **Eval languages** | en, de, es, fr, it, pt, pl, nl, cs, ro, el, uk (12 languages in m-arena-hard-EU) |
| **Held-out languages** | ro, el (zero-shot transfer test) |

### Experiment tracks

| Track | Question | Design |
|---|---|---|
| **A** | Preliminary assessment: english vs. other-languages portion | 90/80/70% English, ~95k samples, [Cohere's fusion-synth-data-s1kx](https://huggingface.co/datasets/CohereLabs/fusion-synth-data-ufb ) only, EU share split equally across 8 languages |
| **B** | What is the effect of adding more data? | Same ratios, ~490k samples, fusion-synth + wildchat, lmsys-chat, oasst2 |
| **C** | Control for A: Is the English regression caused by EU data, or by continued training itself? | 100% English control (no EU data at all), same total samples. If English still regresses, continued SFT may be challenging (hyperparameters?) |
| **D** | Does replaying the base checkpoint's English data reduce forgetting? | Same ratios as Track A, but English data comes from Dolci-Instruct-SFT (the base checkpoint's own training data) instead of new English data |
| **E** | Does Dolci replay scale with more data? | Same ratios as Track D, but ~490k samples instead of 94.7k (Dolci English + same EU sources) |

### Full experiment matrix

| Exp. | En/EU | EU % | Samples | ELO LMArena† | ELO ComparIA‡ | m-arena-hard-EU WR% | arena-hard (en) WR% |
|---|---|---|---|---|---|---|---|
| **Instruct-SFT (ours)** | — | — | — | 766±54 | 247±40 | 50% (ref) | 50% (ref) |
| **A1-90en** | 90/10 | 1.25% | 94.7k | 613±89 | 224±40 | **54.8%** | 14.1% |
| **A2-80en** | 80/20 | 2.5% | 93.6k | 600±110 | 233±37 | **57.2%** | 12.4% |
| **A3-70en** | 70/30 | 3.75% | 91.7k | **709±48** | **235±39** | **58.8%** | 13.3% |
| **B1-90en** | 90/10 | 1.25% | 491k | 633±75 | — | 53.6% | 13.0% |
| **B2-80en** | 80/20 | 2.5% | 473k | 595±136 | — | 52.9% | 14.2% |
| **C0-100en** | 100/0 | — | 94.7k | 524±142 | — | 48.9% | 11.8% |
| **D1-90en** | 90/10 | 1.25% | 94.7k | **755±50** | — | **63.4%** | **54.6%** |
| **D2-80en** | 80/20 | 2.5% | 93.6k | **777±41** | — | **62.0%** | **54.6%** |
| **D3-70en** | 70/30 | 3.75% | 91.7k | 729±90 | — | **63.5%** | **54.3%** |
| **E1-90en** | 90/10 | 1.25% | 491k | **808±51** | — | **59.9%** | **57.0%** |
| **E2-80en** | 80/20 | 2.5% | 474k | 661±130 | — | **56.6%** | **58.2%** |
| **E3-70en** | 70/30 | 3.75% | 455k | 688±79 | — | 53.1% | **58.6%** |

**Sample counts**: Fusion-synth has 94,721 rows across 10 languages, setting the dataset size for Tracks A/D. At 90/10, Czech (1,295 available) fits within its 1.25% share → 94.7k samples. At 80/20 and 70/30, Czech and Dutch (2,800) are capped below their required shares, reducing totals to 93.6k and 91.7k. Track A and D have identical EU distributions — only the English source differs (fusion-synth vs Dolci replay). C0 (100% English) = exactly 94.7k. All sampling is random (not sequential) with a fixed seed (42) for reproducibility.

**~490k** (Track B) = ~5x Track A to test data scaling. Target is 500k, but Czech (1,295) and Dutch (2,800) are capped at what's available, giving actual totals of ~491k (B1) and ~473k (B2).

Winrate = our model vs instruct baseline. 50% = parity. >50% = our model wins.

†**ELO LMArena**: Bradley-Terry, 100 bootstraps, 2.1k battles. Balanced at 200 battles/language (el: 65, ro: 55 — capped at availability), 12 EU languages.

‡**ELO ComparIA**: Bradley-Terry, 100 bootstraps, 20k battles. All languages, predominantly French (~92%).

### Track A: Per-language winrate (m-arena-hard-EU)

`[T]` = trained language, `[H]` = held-out (zero-shot), "uk" = Ukrainian.

| Language | A1-90en | A2-80en | A3-70en |
|---|---:|---:|---:|
| en | 13.2% | 16.0% | 13.6% |
| de [T] | 63.9% | 66.3% | **69.6%** |
| es [T] | 62.6% | **68.4%** | **71.6%** |
| fr [T] | 46.2% | 51.4% | 53.4% |
| it [T] | 51.9% | 60.4% | 59.3% |
| pt [T] | 60.2% | 60.6% | **66.8%** |
| pl [T] | 59.6% | 60.7% | 63.4% |
| nl [T] | 65.1% | 64.3% | 65.1% |
| cs [T] | **73.8%** | **74.2%** | **74.3%** |
| ro [H] | 60.2% | 64.4% | 64.9% |
| el [H] | 51.3% | 50.9% | 49.8% |
| uk | 52.0% | 48.6% | 54.0% |

### Per-language scatter plot (A1 vs A2 vs A3)

Each dot is one language. X-axis is the language, Y-axis is winrate vs baseline.

![Per-language comparison](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/per_language_comparison.png?raw=true)

### Track B/C: Per-language winrate (m-arena-hard-EU)

| Language | B1-90en | B2-80en | C0-100en |
|---|---:|---:|---:|
| en | 14.2% | 14.3% | 15.7% |
| de [T] | 63.6% | 64.0% | 51.7% |
| es [T] | 66.5% | 60.6% | 55.9% |
| fr [T] | 44.2% | 41.5% | 41.3% |
| it [T] | 53.3% | 51.9% | 42.0% |
| pt [T] | 60.3% | 58.0% | 48.4% |
| pl [T] | 56.8% | 57.3% | 48.6% |
| nl [T] | 57.9% | 60.7% | 59.2% |
| cs [T] | **71.7%** | **71.0%** | **61.9%** |
| ro [H] | 58.0% | 55.9% | 61.2% |
| el [H] | 51.9% | 53.8% | 54.2% |
| uk | 45.5% | 46.5% | 46.4% |

### Track D: Per-language winrate (m-arena-hard-EU)

| Language | D1-90en | D2-80en | D3-70en |
|---|---:|---:|---:|
| en | **54.4%** | **56.6%** | **57.7%** |
| de [T] | **67.5%** | **70.5%** | **75.8%** |
| es [T] | **67.1%** | **71.9%** | **72.0%** |
| fr [T] | 57.2% | **60.6%** | **63.9%** |
| it [T] | 59.4% | **60.2%** | **63.0%** |
| pt [T] | **70.7%** | **65.7%** | **70.0%** |
| pl [T] | 59.1% | 59.2% | **60.7%** |
| nl [T] | **68.4%** | **62.5%** | **61.2%** |
| cs [T] | **73.8%** | **70.0%** | **71.9%** |
| ro [H] | **67.7%** | **65.8%** | **64.1%** |
| el [H] | **62.0%** | 51.2% | 51.5% |
| uk | 54.1% | 49.9% | 50.2% |

### Track E: Per-language winrate (m-arena-hard-EU)

| Language | E1-90en | E2-80en | E3-70en |
|---|---:|---:|---:|
| en | **60.3%** | **58.3%** | **57.6%** |
| de [T] | **71.4%** | **66.9%** | 59.9% |
| es [T] | **66.3%** | **61.9%** | 58.2% |
| fr [T] | **61.3%** | 41.2% | 39.2% |
| it [T] | **70.3%** | 53.0% | 41.3% |
| pt [T] | **67.4%** | 58.5% | 56.3% |
| pl [T] | 56.7% | 54.8% | 57.4% |
| nl [T] | 56.5% | 57.3% | 59.2% |
| cs [T] | **68.6%** | **70.3%** | **68.4%** |
| ro [H] | 48.8% | 54.9% | 49.1% |
| el [H] | 46.1% | 49.9% | 43.8% |
| uk | 45.3% | 53.2% | 47.2% |

### Findings

**Track A** (English/EU ratio):
1. **More EU data helps marginally**: A3 (30% EU) = 58.8% > A2 (20%) = 57.2% > A1 (10%) = 54.8%
2. **English regression**: models drop to ~13% winrate on arena-hard (~-37pp), roughly constant regardless of EU ratio
3. **ELO (balanced, w/ en)**: A3 (709) approaches baseline (766). A1 (613) and A2 (600) remain below
4. **Transfer**: Romanian (60-65%) transfers well, Greek (different script, 50-51%) does not
5. **Czech** performs well (74%) — underrepresented in baseline?
6. **French barely moves** (46-53%)

**Track B** (data scaling):
7. **More data does not help**: B1 (53.6%) and B2 (52.9%) perform worse than Track A counterparts despite ~5× more data
8. **English regression persists** at ~13-14%

**Track C** (English-only control):
9. **Continued SFT itself causes forgetting**: C0-100en drops English to 11.8% despite having no EU data at all
10. **C0 slightly hurts EU too**: 48.9% overall (below parity)

**Track D** (Dolci English replay):
11. **Dolci replay preserves English across all ratios**: D1 (54.6%), D2 (54.6%), D3 (54.3%) all maintain English
12. **D2-80en has the best ELO (777±41)**: Actually *exceeds* the baseline (766±54) — the only model to do so
13. **More EU data improves EU languages within Track D**: D3 leads on de (75.8%), es (72.0%), fr (63.9%), it (63.0%)
14. **D3 ELO (729±90) is weaker with wider CI**: Despite good per-language scores, balanced ELO drops
15. **Greek transfer not robust**: D1's Greek gain (62.0%) does not replicate in D2 (51.2%) or D3 (51.5%)

**Track E** (Dolci replay at scale):
16. **E1-90en achieves the highest ELO (808±51)**: Exceeds both the baseline (766±54) and D2 (777±41). English improves to 57.0% on arena-hard, and 60.3% per-language.
17. **E1 m-arena-hard-EU (59.9%)** is lower than D1 (63.4%): Per-language, E1 is stronger on en (60.3% vs 54.4%), fr (61.3% vs 57.2%), it (70.3% vs 59.4%), but weaker on held-out languages ro (48.8% vs 67.7%), el (46.1% vs 62.0%), uk (45.3% vs 54.1%).
18. **E2 (56.6%) and E3 (53.1%) m-arena-hard-EU** are below E1 (59.9%) and their Track D counterparts D2 (62.0%) and D3 (63.5%). French: E2 41.2%, E3 39.2% (vs E1 61.3%). Italian: E3 41.3% (vs E1 70.3%). Czech stays at 68-70% across all three.
19. **E2 ELO (661±130) and E3 ELO (688±79)** are below E1 (808±51) and the baseline (766±54). English arena-hard: E2 58.2%, E3 58.6% (comparable to E1 57.0%).

### Code

- Training configs: [`oellm/configs/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/configs)
- Evaluation scripts: [`oellm/evaluations/benchmarks/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/evaluations/benchmarks)
- Full results: [`oellm/experiments/multilingual_eu/results/multilingual_eu_results.md`](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/multilingual_eu_results.md)
