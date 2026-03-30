## Multilingual SFT: Track A–E results

We're investigating how to add multilingual capabilities to OLMo-3-7B without catastrophic forgetting of English. Starting from our reproduced instruct SFT checkpoint, we continue training on mixtures of English + EU-language data at varying ratios.

### Motivation: Dolci-Instruct-SFT is 93% English

The base checkpoint's training data (Dolci-Instruct-SFT: 2.15M samples, Dolci-Think-SFT: 2.27M samples) is ~93-94% English with a long tail of ~160 languages, none exceeding ~0.2%. This motivates our experiments: can we improve EU language performance by mixing in translated data during continued SFT?

![Dolci-Instruct-SFT language counts](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/dolci_distribution/results/dolci_language_counts.png?raw=true)

![Dolci-Instruct-SFT language percentage](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/dolci_distribution/results/dolci_language_distribution.png?raw=true)

### Setup

| | |
|---|---|
| **Base checkpoint** | OLMo-3-7B-Instruct-SFT (reproduced, [details](https://github.com/allenai/open-instruct/issues/1352#issuecomment-2823953936)) |
| **Judge (winrate)** | Qwen3-30B-A3B-Instruct-2507 (VLLM, both orderings, 8k max tokens) |
| **Judge (Elo)** | Qwen3-30B-A3B-Instruct-2507 (Q3) and Qwen3.5-27B (Q3.5, via vllm serve) |
| **Benchmarks** | m-arena-hard-EU (6000 prompts, 12 EU languages), arena-hard (500 prompts, English) |
| **Training languages** | en + de, es, fr, it, pt, pl, nl, cs (ratio varies by experiment) |
| **Eval languages** | en, de, es, fr, it, pt, pl, nl, cs, ro, el, uk (12 languages in m-arena-hard-EU) |
| **Held-out languages** | ro, el (zero-shot transfer test — not in training data) |

### Experiment tracks

| Track | Question | Design |
|---|---|---|
| **A** | Does the English/EU ratio matter? | 90/80/70% English, 94.7k samples (from fusion-synth), EU share split equally across 8 languages |
| **B** | Does more diverse data help? | Same ratios, ~490k samples, synthetic + wildchat, lmsys-chat, oasst2 |
| **C** | Is the English regression caused by EU data, or by continued training itself? | 100% English control (no EU data at all), same total samples. If English still regresses, the problem is continued SFT, not the EU data. |
| **D** | Does replaying the base checkpoint's English data reduce forgetting? | Same ratios as Track A, but English data comes from Dolci-Instruct-SFT (the base checkpoint's own training data) instead of new English data |
| **E** | Does Dolci replay scale with more data? | Same ratios as Track D, but ~490k samples instead of 94.7k (Dolci English + same EU sources) |

### Full experiment matrix

| Exp. | En/EU | Samples | Elo LMArena† (Q3) | Elo LMArena† (Q3.5) | Elo LMArena w/o en (Q3.5) | Elo ComparIA‡ (Q3) | m-arena-hard-EU WR% | arena-hard (en) WR% |
|---|---|---|---|---|---|---|---|---|
| **Baseline** | — | — | 766±54 | 741±9 | — | 247±40 | 50% | 50% |
| **A1-90en** | 90/10 | 94.7k | 613±89 | 709±10 | 261±35 | 224±40 | **54.8%** | 14.1% |
| **A2-80en** | 80/20 | 93.6k | 600±110 | 700±10 | 242±29 | 233±37 | **57.2%** | 12.4% |
| **A3-70en** | 70/30 | 91.7k | **709±48** | 708±10 | 235±33 | **235±39** | **58.8%** | 13.3% |
| **B1-90en** | 90/10 | 491k | 633±75 | 701±11 | 193±34 | 604±4 | 53.6% | 13.0% |
| **B2-80en** | 80/20 | 473k | 595±136 | 729±11 | 206±30 | 582±5 | 52.9% | 14.2% |
| **C0-100en** | 100/0 | 94.7k | 524±142 | 690±12 | 163±41 | 540±5 | 48.9% | 11.8% |
| **D1-90en** | 90/10 | 94.7k | **755±50** | **748±10** | 190±39 | **716±4** | **63.4%** | **54.6%** |
| **D2-80en** | 80/20 | 93.6k | **777±41** | **748±9** | 219±32 | **739±3** | **62.0%** | **54.6%** |
| **D3-70en** | 70/30 | 91.7k | 729±90 | **748±8** | 195±38 | **760±3** | **63.5%** | **54.3%** |
| **E1-90en** | 90/10 | 491k | **808±51** | **764±8** | **258±29** | **765±3** | **59.9%** | **57.0%** |
| **E2-80en** | 80/20 | 474k | 661±130 | **755±8** | **253±33** | 600±4 | **56.6%** | **58.2%** |
| **E3-70en** | 70/30 | 455k | 688±79 | **754±10** | 212±36 | 596±4 | 53.1% | **58.6%** |

**Sample counts**: Fusion-synth has 94,721 rows across 10 languages, setting the dataset size for Tracks A/D. At 90/10, Czech (1,295 available) fits within its 1.25% share → 94.7k samples. At 80/20 and 70/30, Czech and Dutch (2,800) are capped below their required shares, reducing totals to 93.6k and 91.7k. Track A and D have identical EU distributions — only the English source differs (fusion-synth vs Dolci replay). C0 (100% English) = exactly 94.7k. All sampling is random (not sequential) with a fixed seed (42) for reproducibility.

**~490k** (Track B) = ~5× Track A to test data scaling. Target is 500k, but Czech (1,295) and Dutch (2,800) are capped at what's available, giving actual totals of ~491k (B1) and ~473k (B2).

Winrate = our model vs instruct baseline. 50% = parity. >50% = our model wins.

†**Elo LMArena**: Bradley-Terry, 100 bootstraps, ~2.1k battles. Balanced at 200 battles/language, 12 EU languages. Q3 = Qwen3-30B-A3B-Instruct-2507, Q3.5 = Qwen3.5-27B (dense). Q3.5 CIs are much tighter (±7-11 vs ±41-142).

**Elo w/o en**: Same as † but excluding English battles (11 languages, ~1.9k battles). Only measured with Q3.5 judge.

‡**ComparIA**: Bradley-Terry, 100 bootstraps, 20k battles. All languages, predominantly French (~92%).

### Track A: Per-language winrate (m-arena-hard-EU, Q3 judge)

Winrate = our model vs instruct baseline (head-to-head). 50% = parity. Judge: Qwen3-30B-A3B-Instruct-2507.

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

### Track B/C: Per-language winrate (m-arena-hard-EU, Q3 judge)

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

### Track D: Per-language winrate (m-arena-hard-EU, Q3 judge)

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

### Track E: Per-language winrate (m-arena-hard-EU, Q3 judge)

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

### Per-language winrate from Elo battles (Qwen3.5-27B judge, LMArena)

Raw winrate of our model vs LMArena arena opponents, computed from the judge cache of the balanced Elo runs (200 battles/language, 12 EU languages). Unlike m-arena-hard-EU (which is our model vs baseline), this is our model vs the full arena field (GPT-4o, Claude, Gemini, etc.), so winrates are low (~10-40%) for a 7B model.

| Model | en | de | es | fr | it | pt | pl | nl | cs | ro | el | uk | ALL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A2-80en** | 15.8 | 7.5 | 20.0 | 9.5 | 21.5 | 15.0 | 12.5 | 15.8 | 10.8 | 16.4 | 6.2 | 11.5 | 13.8 |
| **A3-70en** | 16.2 | 9.0 | 16.8 | 13.8 | 19.8 | 17.0 | 15.5 | 13.8 | 13.0 | 20.9 | 7.7 | 9.2 | 14.4 |
| **D1-90en** | **41.8** | 6.5 | 19.2 | 13.5 | 20.5 | 16.0 | 13.0 | 19.5 | 12.5 | **30.9** | 5.4 | 11.2 | 17.4 |
| **D2-80en** | **35.0** | 9.8 | 21.5 | 11.8 | 24.2 | 13.5 | 11.8 | 16.2 | 16.8 | **29.1** | 6.2 | 14.0 | 17.4 |
| **D3-70en** | **40.2** | 9.2 | 22.0 | 10.0 | 21.5 | 19.8 | 14.5 | 16.2 | 10.0 | **28.2** | 6.2 | 10.5 | 17.3 |
| **E1-90en** | **38.2** | **13.5** | 21.0 | 13.8 | **25.5** | 16.2 | 17.0 | **20.0** | 10.5 | **29.1** | 9.2 | 11.8 | **18.7** |
| **E3-70en** | **38.8** | 11.2 | 17.2 | 14.8 | 21.8 | 16.8 | 14.8 | 18.5 | 12.8 | **28.2** | 8.5 | 12.5 | 17.9 |

D-track's advantage over A-track comes almost entirely from **English** (19–24pp gap) and **Romanian** (7–13pp gap). On other trained EU languages, A and D perform within ±5pp. This explains why D-track w/o English Elo is lower: remove English, and D-track loses its main advantage.

### Key findings

1. **Dolci replay (Track D) preserves English**: D-track maintains ~55% English arena-hard winrate (vs A-track's ~13%). D-track Elo is stable at 748 regardless of EN/EU ratio — the English source matters, the ratio doesn't.
2. **D-track advantage is entirely English**: Per-language Elo analysis shows A and D within ±5pp on all trained EU languages. The ~48 Elo gap (748 vs 700) comes from English (19-24pp) and Romanian transfer (7-13pp).
3. **Scaling helps modestly (Track E)**: E1 achieves the highest Elo (764 Q3.5). E2/E3 drop on French (41%/39% vs E1's 61%) — more EU data at scale hurts some languages.
4. **More diverse data doesn't help (Track B)**: B-track (5× more data from wildchat/lmsys/oasst2) scores below A-track on m-arena-hard-EU despite 5× samples.
5. **Continued SFT itself causes English forgetting (Track C)**: C0 (100% English, no EU data) still drops English to 12%. C0 Elo (690) is only 10-20 points below A-track (700-709).
6. **Q3.5 Elo tiers**: E-track (754-764) > D-track (748) > baseline (741) > A/B (700-729) > C0 (690). CIs of ±8-12 (vs Q3's ±41-142) make these separations reliable.

### Next steps

**Incoming: high-quality Dolci translations** (gemma3-27b-it, [tracking issue](https://github.com/OpenEuroLLM/Taskboard/issues/193)):
- **Dolci-Think-SFT-7B** → 7 languages (cs, de, it, fr, fi, es, sv) — ETA end of March/early April
- **Dolci-Instruct-SFT** → same 7 languages — AI Sweden starting soon

These are significantly higher quality than our current NLLB-200-distilled-600M translations. Since the per-language analysis shows EU language winrates are modest (6-22% vs arena field), better translations may be the bottleneck.

**Track F (planned)**: High-quality Dolci-Instruct translations with Dolci replay
- Same design as Track D (Dolci English replay), but EU data = gemma3-27b-it translations
- Languages: cs, de, it, fr, fi, es, sv (7 high-quality) + pl, nl (NLLB, 2)
- Ratio: 90/10 only (D-track showed ratio doesn't matter with Dolci replay)
- **Purpose**: Isolate translation quality effect. If F1 >> D1 on EU languages, translation quality was the bottleneck.

**Track G (planned)**: High-quality Dolci-Think translations
- Same as F but using Dolci-Think-SFT translations
- **Purpose**: Compare Think vs Instruct as multilingual data source.

### Code

- Training configs: [`oellm/configs/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/configs)
- Evaluation scripts: [`oellm/evaluations/benchmarks/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/evaluations/benchmarks)
