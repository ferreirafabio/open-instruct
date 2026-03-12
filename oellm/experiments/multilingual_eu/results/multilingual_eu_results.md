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
| **Judge** | Qwen3-30B-A3B-Instruct-2507 (VLLM, winrate mode, both orderings) |
| **Benchmarks** | m-arena-hard-EU (6000 prompts, 12 EU languages), arena-hard (500 prompts, English) |
| **Training languages** | en + de, es, fr, it, pt, pl, nl, cs (ratio varies by experiment) |
| **Eval languages** | en, de, es, fr, it, pt, pl, nl, cs, ro, el, uk (12 languages in m-arena-hard-EU) |
| **Held-out languages** | ro, el (zero-shot transfer test — not in training data) |

### Experiment tracks

| Track | Question | Design |
|---|---|---|
| **A** | Does the English/EU ratio matter? | 90/80/70% English, 94.7k samples (from fusion-synth), EU share split equally across 8 languages |
| **B** | Does more diverse data help? | Same ratios, ~490k samples, synthetic + organic (wildchat, lmsys-chat, oasst2) |
| **C** | Is the English regression caused by EU data, or by continued training itself? | 100% English control (no EU data at all), same total samples. If English still regresses, the problem is continued SFT, not the EU data. |
| **D** | Does replaying the base checkpoint's English data reduce forgetting? | Same as Track B ratios, but English data comes from Dolci-Instruct-SFT (the base checkpoint's own training data) instead of new English data |

### Full experiment matrix

| Experiment | En/EU ratio | Per-EU-lang % | Total samples | ELO | m-arena-hard-EU WR% | arena-hard WR% |
|---|---|---|---|---|---|---|
| **Instruct baseline** | — | — | — | 1154.2 ± 114.4 | — | — |
| **A1-90en** | 90/10 | 1.25% each | 94.7k | 686.4 ± 69.8 | **54.8%** (+4.8pp) | 14.1% (-35.9pp) |
| **A2-80en** | 80/20 | 2.5% each | 94.7k | _pending_ | **57.2%** (+7.2pp) | 12.4% (-37.6pp) |
| **A3-70en** | 70/30 | 3.75% each | 94.7k | _pending_ | **58.8%** (+8.8pp) | 13.3% (-36.7pp) |
| **B1-90en** | 90/10 | 1.25% each | 491k | - | _pending_ | _pending_ |
| **B2-80en** | 80/20 | 2.5% each | 473k | - | _pending_ | _pending_ |
| **C0-100en** | 100/0 | — | 94.7k | - | _pending_ | _pending_ |
| **D1-90en** | 90/10 (replay EN) | 1.25% each | 94.7k | - | _pending_ | _pending_ |

**94.7k** = total row count of the fusion-synth dataset (94,721 samples across 10 languages). Track A uses fusion-synth as its primary multilingual source, which covers de/es/fr/it/pt well (~8-10k each), while WildChat and lmsys-chat fill gaps for pl/nl/cs. No upsampling; A2/A3 cap Czech at its available 1,295 samples, so actual totals are slightly below 94.7k. C0 and D1 use the same 94.7k for direct comparability.

**~490k** (Track B) = ~5× Track A to test data scaling. Target is 500k, but Czech (1,295) and Dutch (2,800) are capped at what's available, giving actual totals of ~491k (B1) and ~473k (B2).

Winrate = our model vs instruct baseline. 50% = parity. >50% = our model wins.

### Track A: Per-language winrate (m-arena-hard-EU)

`[T]` = trained language, `[H]` = held-out (zero-shot), `[E]` = English. "uk" = Ukrainian.

_Re-evaluating with swap_mode=both (previous results used fixed ordering, which underestimates winrates by ~35pp due to position bias)._

| Language | A1-90en | A2-80en | A3-70en |
|---|---:|---:|---:|
| _pending_ | | | |

### Key findings (Track A)

1. **More EU data helps**: A3 (30% EU) = 58.8% > A2 (20%) = 57.2% > A1 (10%) = 54.8% on m-arena-hard-EU.
2. **Severe English regression**: All models drop to ~13% winrate on arena-hard (~-37pp). The degradation is roughly constant regardless of EU ratio, suggesting continued training itself causes forgetting.
3. **Baseline ELO is much higher**: 1154 vs 686 for A1. The multilingual training hurts absolute competitiveness on the LMArena leaderboard, driven by English regression.
4. Per-language breakdown pending.

### Next steps

- **Track C** (C0-100en, English-only control) will isolate whether English regression comes from EU data or from continued training itself
- **Track B** (B1/B2, more data + organic sources) will show if scale and data diversity help
- **ELO ratings**: Running arena-anchored Bradley-Terry ELO estimation on EU languages (pending)

### Code

- Training configs: [`oellm/configs/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/configs)
- Evaluation scripts: [`oellm/evaluations/benchmarks/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/evaluations/benchmarks)
