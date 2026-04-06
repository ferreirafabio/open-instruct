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
| **F** | How do extreme English/EU ratios affect performance with Dolci replay? | 100/75/50/25/0% English, 500k samples, 6 EU languages (de,es,fr,it,pt,pl — drop cs/nl due to data scarcity at scale), Dolci replay |

### Full experiment matrix

| Exp. | En/EU | N | Elo† | Elo en† | Elo w/o en† | EU WR% (Q3) | en WR% (Q3) |
|---|---|---|---|---|---|---|---|
| **Baseline** | — | — | 741±9 | **950±21** | 722±10 | 50% | 50% |
| **A1-90en** | 90/10 | 94.7k | 702±10 | 771±32 | 692±10 | **54.8%** | 14.1% |
| **A2-80en** | 80/20 | 93.6k | 704±11 | 769±30 | 703±11 | **57.2%** | 12.4% |
| **A3-70en** | 70/30 | 91.7k | 713±10 | 766±29 | 689±11 | **58.8%** | 13.3% |
| **B1-90en** | 90/10 | 491k | 720±9 | 789±26 | 708±10 | 53.6% | 13.0% |
| **B2-80en** | 80/20 | 473k | 722±9 | 797±26 | 722±10 | 52.9% | 14.2% |
| **C0-100en** | 100/0 | 94.7k | 670±11 | 791±29 | 681±11 | 48.9% | 11.8% |
| **D1-90en** | 90/10 | 94.7k | **751±8** | **942±20** | 716±11 | **63.4%** | **54.6%** |
| **D2-80en** | 80/20 | 93.6k | **751±8** | **956±20** | **725±12** | **62.0%** | **54.6%** |
| **D3-70en** | 70/30 | 91.7k | **753±9** | **963±21** | **731±11** | **63.5%** | **54.3%** |
| **E1-90en** | 90/10 | 491k | **758±9** | **965±21** | **740±9** | **59.9%** | **57.0%** |
| **E2-80en** | 80/20 | 474k | **759±8** | **931±24** | **725±9** | **56.6%** | **58.2%** |
| **E3-70en** | 70/30 | 455k | **751±9** | **940±22** | **726±9** | 53.1% | **58.6%** |
| **F1-100en** | 100/0 | 500k | 728±9 | **954±22** | 695±11 | 52.2% | **57.6%** |
| **F2-75en** | 75/25 | 486k | **747±10** | **947±22** | 711±10 | 51.4% | **59.7%** |
| **F3-50en** | 50/50 | 453k | 726±9 | **935±21** | 700±12 | 44.7% | **55.9%** |
| **F4-25en** | 25/75 | 359k | 739±9 | **912±22** | 685±11 | 43.3% | **57.2%** |
| **F5-0en** | 0/100 | 234k | 677±11 | 762±31 | 680±11 | 36.6% | 9.7% |

† Elo: Qwen3.5-27B judge, LMArena Bradley-Terry, 100 bootstraps, 200 battles/lang. "en" = English-only (200 battles). "w/o en" = 11 non-English languages.

EU WR% / en WR% = winrate vs baseline (Qwen3-30B-A3B judge). 50% = parity.

### Elo with Qwen3-30B-A3B judge (Q3)

| Exp. | En/EU | N | Elo LMArena | Elo ComparIA |
|---|---|---|---|---|
| **Baseline** | — | — | 766±54 | 247±40 |
| **A1-90en** | 90/10 | 94.7k | 613±89 | 224±40 |
| **A2-80en** | 80/20 | 93.6k | 600±110 | 233±37 |
| **A3-70en** | 70/30 | 91.7k | **709±48** | **235±39** |
| **B1-90en** | 90/10 | 491k | 633±75 | 604±4 |
| **B2-80en** | 80/20 | 473k | 595±136 | 582±5 |
| **C0-100en** | 100/0 | 94.7k | 524±142 | 540±5 |
| **D1-90en** | 90/10 | 94.7k | **755±50** | **716±4** |
| **D2-80en** | 80/20 | 93.6k | **777±41** | **739±3** |
| **D3-70en** | 70/30 | 91.7k | 729±90 | **760±3** |
| **E1-90en** | 90/10 | 491k | **808±51** | **765±3** |
| **E2-80en** | 80/20 | 474k | 661±130 | 600±4 |
| **E3-70en** | 70/30 | 455k | 688±79 | 596±4 |

LMArena: 200 battles/lang, 12 EU languages. ComparIA: 20k battles, mostly French (~92%). Both: Bradley-Terry, 100 bootstraps.

**Sample counts**: Fusion-synth has 94,721 rows across 10 languages, setting the dataset size for Tracks A/D. At 90/10, Czech (1,295 available) fits within its 1.25% share → 94.7k samples. At 80/20 and 70/30, Czech and Dutch (2,800) are capped below their required shares, reducing totals to 93.6k and 91.7k. Track A and D have identical EU distributions — only the English source differs (fusion-synth vs Dolci replay). C0 (100% English) = exactly 94.7k. All sampling is random (not sequential) with a fixed seed (42) for reproducibility.

**~490k** (Track B) = ~5× Track A to test data scaling. Target is 500k, but Czech (1,295) and Dutch (2,800) are capped at what's available, giving actual totals of ~491k (B1) and ~473k (B2).

### Per-language winrate from Elo battles (Q3.5 judge, LMArena)

Winrate of our model vs LMArena arena opponents (GPT-4o, Claude, Gemini, etc.), from the balanced Elo judge cache (200 battles/language). Winrates are low (~10-40%) as expected for a 7B model vs frontier models.

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

### Per-language winrate vs baseline (m-arena-hard-EU, Q3 judge)

Winrate = our model vs instruct baseline (head-to-head). 50% = parity. Judge: Qwen3-30B-A3B-Instruct-2507.

`[T]` = trained language, `[H]` = held-out (zero-shot). "uk" = Ukrainian.

**Track A:**

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

**Track B/C:**

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

**Track D:**

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

**Track E:**

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

### Key findings

1. **Dolci replay (Track D) preserves English**: D-track maintains ~55% English arena-hard winrate (vs A-track's ~13%). D-track Elo is stable at 748 regardless of EN/EU ratio — the English source matters, the ratio doesn't.
2. **D-track advantage holds with and without English**: D-track leads both w/ en (751 vs A-track 702-713) and w/o en (716-731 vs A-track 689-703). Per-language Elo battle analysis shows the gap is largest on English (19-24pp) and Romanian (7-13pp), with ±5pp on other trained EU languages.
3. **Scaling helps modestly (Track E)**: E1 achieves the highest Elo (764 Q3.5). E2/E3 drop on French (41%/39% vs E1's 61%) — more EU data at scale hurts some languages.
4. **More diverse data doesn't help (Track B)**: B-track (5× more data from wildchat/lmsys/oasst2) scores below A-track on m-arena-hard-EU despite 5× samples.
5. **Continued SFT itself causes English forgetting (Track C)**: C0 (100% English, no EU data) still drops English to 12%. C0 Elo (690) is only 10-20 points below A-track (700-709).
6. **Q3.5 Elo tiers** (w/ en): E-track (751-759) > D-track (751-753) > baseline (741) > A/B (702-722) > C0 (670). W/o en: E1 (740) > D3 (731) > D2 (725) ≈ E2/E3 (725-726) > baseline (722) > D1 (716) > B1 (708) > A (689-703) > C0 (681). CIs of ±8-12 (vs Q3's ±41-142) make these separations reliable.

### Next steps

**Incoming: high-quality Dolci translations** (gemma3-27b-it, [tracking issue](https://github.com/OpenEuroLLM/Taskboard/issues/193)):
- **Dolci-Think-SFT-7B** → 7 languages (cs, de, it, fr, fi, es, sv) — ETA end of March/early April
- **Dolci-Instruct-SFT** → same 7 languages — AI Sweden starting soon

These are significantly higher quality than our current NLLB-200-distilled-600M translations. Since the per-language analysis shows EU language winrates are modest (6-22% vs arena field), better translations may be the bottleneck.

**Track F (configs ready)**: Extreme English/EU ratios with Dolci replay at 500k scale
- F1-F5: 100/75/50/25/0% English, 500k samples, Dolci replay
- 6 EU languages (de, es, fr, it, pt, pl) — cs/nl dropped due to data scarcity at 500k scale
- **Purpose**: Map the full English/EU ratio curve. How far can we push EU data before English collapses, even with Dolci replay?

**Track G (planned)**: High-quality Dolci translations with gemma3-27b-it
- Same design as Track D (Dolci English replay), but EU data = gemma3-27b-it translations
- Languages: cs, de, it, fr, fi, es, sv (7 high-quality) + pl, nl (NLLB, 2)
- **Purpose**: Isolate translation quality effect. If G >> D on EU languages, translation quality was the bottleneck.

**Track H (planned)**: High-quality Dolci-Think translations
- Same as G but using Dolci-Think-SFT translations
- **Purpose**: Compare Think vs Instruct as multilingual data source.

### Code

- Training configs: [`oellm/configs/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/configs)
- Evaluation scripts: [`oellm/evaluations/benchmarks/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/evaluations/benchmarks)
