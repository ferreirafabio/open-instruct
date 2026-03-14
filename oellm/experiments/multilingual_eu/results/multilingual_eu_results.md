## Multilingual SFT: Track A results

We're investigating how to add multilingual capabilities to OLMo-3-7B without catastrophic forgetting of English. Starting from our reproduced instruct SFT checkpoint, we continue training on mixtures of English + EU-language data at varying ratios.

### Motivation: Dolci-Instruct-SFT is 93% English

The base checkpoint's training data (Dolci-Instruct-SFT: 2.15M samples, Dolci-Think-SFT: 2.27M samples) is ~93-94% English with a long tail of ~160 languages, none exceeding ~0.2%. This motivates our experiments: can we improve EU language performance by mixing in translated data during continued SFT?

![Dolci-Instruct-SFT language counts](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/dolci_distribution/results/dolci_language_counts.png?raw=true)

![Dolci-Instruct-SFT language percentage](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/dolci_distribution/results/dolci_language_distribution.png?raw=true)

### Setup

| | |
|---|---|
| **Base checkpoint** | OLMo-3-7B-Instruct-SFT (reproduced, [details](https://github.com/allenai/open-instruct/issues/1352#issuecomment-2823953936)) |
| **Judge** | Qwen3-30B-A3B-Instruct-2507 (VLLM, winrate mode, both orderings, 8k max tokens) |
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

| Exp. | En/EU | EU % | Samples | ELO LMArena† | ELO ComparIA‡ | m-arena-hard-EU WR% | arena-hard (en) WR% |
|---|---|---|---|---|---|---|---|
| **Instruct-SFT (ours)** | — | — | — | 766±54 | 247±40 | 50% (ref) | 50% (ref) |
| **A1-90en** | 90/10 | 1.25% | 94.7k | 613±89 | 224±40 | **54.8%** | 14.1% |
| **A2-80en** | 80/20 | 2.5% | 94.7k | 600±110 | 233±37 | **57.2%** | 12.4% |
| **A3-70en** | 70/30 | 3.75% | 94.7k | **709±48** | **235±39** | **58.8%** | 13.3% |
| **B1-90en** | 90/10 | 1.25% | 491k | 633±75 | — | 53.6% | 13.0% |
| **B2-80en** | 80/20 | 2.5% | 473k | 595±136 | — | 52.9% | 14.2% |
| **C0-100en** | 100/0 | — | 94.7k | 524±142 | — | 48.9% | 11.8% |
| **D1-90en** | 90/10 | 1.25% | 94.7k | **755±50** | — | **63.4%** | **54.6%** |

**94.7k** = total row count of the fusion-synth dataset (94,721 samples across 10 languages). Track A uses fusion-synth as its primary multilingual source, which covers de/es/fr/it/pt well (~8-10k each), while WildChat and lmsys-chat fill gaps for pl/nl/cs. No upsampling; A2/A3 cap Czech at its available 1,295 samples, so actual totals are slightly below 94.7k. C0 and D1 use the same 94.7k for direct comparability.

**~490k** (Track B) = ~5× Track A to test data scaling. Target is 500k, but Czech (1,295) and Dutch (2,800) are capped at what's available, giving actual totals of ~491k (B1) and ~473k (B2).

Winrate = our model vs instruct baseline. 50% = parity. >50% = our model wins.

†**ELO LMArena**: Bradley-Terry, 100 bootstraps, 2.1k battles. Balanced at 200 battles/language (el: 65, ro: 55 — capped at availability), 12 EU languages.

‡**ELO ComparIA**: Bradley-Terry, 100 bootstraps, 20k battles. All languages, predominantly French (~92%).

### Track A: Per-language winrate (m-arena-hard-EU)

`[T]` = trained language, `[H]` = held-out (zero-shot). "uk" = Ukrainian.

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

### Track B/C/D: Per-language winrate (m-arena-hard-EU)

| Language | B1-90en | B2-80en | C0-100en | D1-90en |
|---|---:|---:|---:|---:|
| en | 14.2% | 14.3% | 15.7% | **54.4%** |
| de [T] | 63.6% | 64.0% | 51.7% | **67.5%** |
| es [T] | 66.5% | 60.6% | 55.9% | **67.1%** |
| fr [T] | 44.2% | 41.5% | 41.3% | 57.2% |
| it [T] | 53.3% | 51.9% | 42.0% | 59.4% |
| pt [T] | 60.3% | 58.0% | 48.4% | **70.7%** |
| pl [T] | 56.8% | 57.3% | 48.6% | 59.1% |
| nl [T] | 57.9% | 60.7% | 59.2% | **68.4%** |
| cs [T] | **71.7%** | **71.0%** | **61.9%** | **73.8%** |
| ro [H] | 58.0% | 55.9% | 61.2% | **67.7%** |
| el [H] | 51.9% | 53.8% | 54.2% | **62.0%** |
| uk | 45.5% | 46.5% | 46.4% | 54.1% |

### Key findings

**Track A** (English/EU ratio):
1. **More EU data helps**: A3 (30% EU) = 58.8% > A2 (20%) = 57.2% > A1 (10%) = 54.8% on m-arena-hard-EU.
2. **Severe English regression**: All models drop to ~13% winrate on arena-hard (~-37pp). The degradation is roughly constant regardless of EU ratio, suggesting continued training itself causes forgetting.
3. **ELO (balanced, w/ en)**: With language-balanced battles, A3 (709) approaches the baseline (766). A1 (613) and A2 (600) remain below — English regression still hurts on the English portion of battles.
4. **Transfer**: Romanian (60-65%) transfers well despite not being in training data. Greek (50-51%, different script) does not.
5. **Czech** performs best (74%) — likely underrepresented in the baseline.
6. **French barely moves** (46-53%).

**Track B** (data scaling):
7. **More data does not help**: B1 (53.6%) and B2 (52.9%) perform *worse* than their Track A counterparts A1 (54.8%) and A2 (57.2%), despite having ~5× more data. Adding organic sources (wildchat, lmsys-chat) at scale may introduce noise.
8. **English regression persists**: B1 (13.0%) and B2 (14.2%) show the same ~37pp English drop as Track A.

**Track C** (English-only control):
9. **Continued SFT itself causes forgetting**: C0-100en (no EU data at all) still drops English to 11.8% on arena-hard. This confirms the regression is not caused by EU data — it's an artifact of continued training on new English data.
10. **C0 slightly hurts EU languages too**: 48.9% overall EU winrate (below 50% parity), suggesting the new English data actively overwrites some multilingual capability.

**Track D** (Dolci English replay):
11. **Dolci replay preserves English**: D1-90en is the only model that maintains English performance (54.6% arena-hard, vs ~13% for all others). Replaying the base checkpoint's own training data during continued SFT prevents catastrophic forgetting.
12. **D1 achieves best EU scores across nearly every language**: 63.4% overall, with strong gains even in held-out languages — Greek jumps to 62.0% (vs ~50% in Track A) and Romanian to 67.7%.
13. **D1 ELO (755±50) approaches baseline (766±54)**: The only continued-SFT model that nearly matches the base checkpoint's balanced ELO rating.

### Code

- Training configs: [`oellm/configs/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/configs)
- Evaluation scripts: [`oellm/evaluations/benchmarks/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/evaluations/benchmarks)
