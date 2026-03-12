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
| **D** | Does replaying the base checkpoint's English data reduce forgetting? | Same as Track B, but English data comes from Dolci-Instruct-SFT (the base checkpoint's own training data) instead of new English data |

### Full experiment matrix

| Exp. | En/EU | EU-lang % | Samples | ELO† | m-arena-hard-EU WR% | arena-hard WR% |
|---|---|---|---|---|---|---|
| **Baseline** | — | — | — | 688±75 | — | — |
| **A1-90en** | 90/10 | 1.25% | 94.7k | 613±89 | **54.8%** | 14.1% |
| **A2-80en** | 80/20 | 2.5% | 94.7k | 600±110 | **57.2%** | 12.4% |
| **A3-70en** | 70/30 | 3.75% | 94.7k | **709±48** | **58.8%** | 13.3% |
| **B1-90en** | 90/10 | 1.25% | 491k | - | _pending_ | _pending_ |
| **B2-80en** | 80/20 | 2.5% | 473k | - | _pending_ | _pending_ |
| **C0-100en** | 100/0 | — | 94.7k | - | _pending_ | _pending_ |
| **D1-90en** | 90/10 | 1.25% | 94.7k | - | _pending_ | _pending_ |

**94.7k** = total row count of the fusion-synth dataset (94,721 samples across 10 languages). Track A uses fusion-synth as its primary multilingual source, which covers de/es/fr/it/pt well (~8-10k each), while WildChat and lmsys-chat fill gaps for pl/nl/cs. No upsampling; A2/A3 cap Czech at its available 1,295 samples, so actual totals are slightly below 94.7k. C0 and D1 use the same 94.7k for direct comparability.

**~490k** (Track B) = ~5x Track A to test data scaling. Target is 500k, but Czech (1,295) and Dutch (2,800) are capped at what's available, giving actual totals of ~491k (B1) and ~473k (B2).

Winrate = our model vs instruct baseline. 50% = parity. >50% = our model wins.

†**ELO**: Bradley-Terry with 100 bootstraps on LMArena battles, balanced at 200 battles/language (el: 65, ro: 55 — capped at availability), 12 EU languages, 2,120 battles total.

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

### Preliminary Findings (Track A)

1. **Expectations met: more EU data helps marginally with EU language eval**: A3 (30% EU) = 58.8% > A2 (20%) = 57.2% > A1 (10%) = 54.8%
2. **English regression**: models drop to ~13% winrate on arena-hard (~-37pp), which serves as English control test. The degradation is roughly constant regardless of EU ratio (is continued SFT tricky?).
3. **ELO (balanced, w/ en)**: With language-balanced battles, A3 (709) overtakes the baseline (688). A1 (613) and A2 (600) remain below — English regression still hurts on the English portion of battles.
4. **Transfer**: Romanian and Greek are not in train data. Romanian transfers well (60-65%), Greek (different script) does not (50-51%).
5. **Czech** performs well (74%) -> underrepresented in the baseline?
6. **French barely moves** (46-53%)

### Code

- Training configs: [`oellm/configs/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/configs)
- Evaluation scripts: [`oellm/evaluations/benchmarks/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/evaluations/benchmarks)
- Full results: [`oellm/experiments/multilingual_eu/results/multilingual_eu_results.md`](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/multilingual_eu_results.md)
