## Multilingual fine-tuning: Track A results

We're investigating how to add multilingual capabilities to OLMo-3-7B without catastrophic forgetting of English. Starting from our reproduced instruct SFT checkpoint, we continue training on mixtures of English + EU-language data at varying ratios.

### Motivation: Dolci-Instruct-SFT is 93% English

The base checkpoint's training data (Dolci-Instruct-SFT: 2.15M samples, Dolci-Think-SFT: 2.27M samples) is ~93-94% English with a long tail of ~160 languages, none exceeding ~0.2%. This motivates our experiments: can we improve EU language performance by mixing in translated data during continued SFT?

![Dolci-Instruct-SFT language counts](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/dolci_distribution/results/dolci_language_counts.png?raw=true)

![Dolci-Instruct-SFT language percentage](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/dolci_distribution/results/dolci_language_distribution.png?raw=true)

### Setup

| | |
|---|---|
| **Base checkpoint** | OLMo-3-7B-Instruct-SFT (reproduced, [details](https://github.com/allenai/open-instruct/issues/1352#issuecomment-2823953936)) |
| **Judge** | Qwen3-30B-A3B-Instruct-2507 (VLLM, winrate mode, fixed ordering) |
| **Benchmarks** | m-arena-hard-EU (6000 prompts, 12 EU languages), arena-hard (500 prompts, English) |
| **Training languages** | de, es, fr, it, pt, pl, nl, cs |
| **Held-out languages** | ro, el (zero-shot transfer test) |

### Experiment tracks

| Track | Question | Design |
|---|---|---|
| **A** | Does the English/EU ratio matter? | 90/80/70% English, ~95k samples, synthetic data only |
| **B** | Does more diverse data help? | Same ratios, ~490k samples, synthetic + organic (wildchat, lmsys-chat, oasst2) |
| **C** | Is the English regression caused by EU data, or by continued training itself? | 100% English control (no EU data at all), same total samples. If English still regresses, the problem is continued SFT, not the EU data. |
| **D** | Does replaying the base checkpoint's English data reduce forgetting? | Same as Track B ratios, but English data comes from Dolci-Instruct-SFT (the base checkpoint's own training data) instead of new English data |

### Full experiment matrix

| Experiment | En/EU ratio | Total samples | ELO | m-arena-hard-EU WR% | arena-hard WR% |
|---|---|---|---|---|---|
| **A1-90en** | 90/10 | 94.7k | _pending_ | **57.2%** (+7.2pp) | 15.7% (-34.3pp) |
| **A2-80en** | 80/20 | 94.7k | _pending_ | **58.5%** (+8.5pp) | 13.7% (-36.3pp) |
| **A3-70en** | 70/30 | 94.7k | _pending_ | **59.0%** (+9.0pp) | 14.3% (-35.7pp) |
| **B1-90en** | 90/10 | 491k | - | _pending_ | _pending_ |
| **B2-80en** | 80/20 | 473k | - | _pending_ | _pending_ |
| **C0-100en** | 100/0 | 94.7k | - | _pending_ | _pending_ |
| **D1-90en** | 90/10 (replay EN) | 94.7k | - | _pending_ | _pending_ |

Winrate = our model vs instruct baseline. 50% = parity. >50% = our model wins.

### Track A: Per-language winrate (m-arena-hard-EU)

`[T]` = trained language, `[H]` = held-out (zero-shot), `[E]` = English. "uk" = Ukrainian.

| Language | A1-90en | A2-80en | A3-70en |
|---|---:|---:|---:|
| en [E] | 14.9% | 13.9% | 12.9% |
| de [T] | 65.7% | 68.5% | **71.0%** |
| es [T] | 65.7% | **72.2%** | **72.2%** |
| fr [T] | 49.8% | 49.8% | 52.1% |
| it [T] | 55.1% | 58.5% | 59.9% |
| pt [T] | 62.9% | 69.6% | **70.0%** |
| pl [T] | 60.6% | 63.4% | **67.0%** |
| nl [T] | 67.3% | 65.1% | 67.6% |
| cs [T] | **74.5%** | **76.2%** | **75.7%** |
| ro [H] | 64.8% | 65.5% | 61.6% |
| el [H] | 54.3% | 48.5% | 47.1% |
| uk | 51.1% | 50.3% | 51.1% |

### Per-language scatter plot (A1 vs A2 vs A3)

Each dot is one language. X-axis is the language, Y-axis is winrate vs baseline.

![Per-language comparison](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/per_language_comparison.png?raw=true)

### Key findings (Track A)

1. **More EU data helps marginally**: A3 (30% EU) = 59.0% > A2 (20%) = 58.5% > A1 (10%) = 57.2%
2. **Severe English regression**: All models drop to ~14% winrate on arena-hard (~-35pp). The degradation is roughly constant regardless of EU ratio, suggesting continued training itself causes forgetting.
3. **Zero-shot transfer**: Romanian (Latin script, Romance family) transfers well (62-66%). Greek (different script) does not (47-54%).
4. **Czech is the biggest winner** (74-76%), likely because it's underrepresented in the baseline.
5. **French barely moves** (~50%) despite being a trained language.

### Next steps

- **Track C** (C0-100en, English-only control) will isolate whether English regression comes from EU data or from continued training itself
- **Track B** (B1/B2, more data + organic sources) will show if scale and data diversity help
- **ELO ratings**: Running arena-anchored Bradley-Terry ELO estimation (pending)

### Code

- Training configs: [`oellm/configs/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/configs)
- Evaluation scripts: [`oellm/evaluations/benchmarks/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/evaluations/benchmarks)
- Full results: [`oellm/experiments/multilingual_eu/results/multilingual_eu_results.md`](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/multilingual_eu_results.md)
