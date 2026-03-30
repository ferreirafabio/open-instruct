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
| **Baseline** | — | — | 766±54 | — | — | 247±40 | 50% | 50% |
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

**A-track vs D-track difference** (positive = A better):

| Comparison | en | de | es | fr | it | pt | pl | nl | cs | ro | el | uk | ALL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A2 − D1 | **-26.0** | +1.0 | +0.8 | -4.0 | +1.0 | -1.0 | -0.5 | -3.8 | -1.8 | -14.5 | +0.8 | +0.3 | -3.6 |
| A3 − D2 | **-18.7** | -0.8 | -4.7 | +2.0 | -4.5 | +3.5 | +3.8 | -2.5 | -3.8 | -8.2 | +1.5 | -4.8 | -3.0 |

D-track's advantage over A-track comes almost entirely from **English** (26–19pp gap) and **Romanian** (8–15pp gap). On other EU languages, A and D perform within ±5pp of each other. This explains why D-track w/o English Elo is lower than expected: remove English, and D-track loses its main advantage.

### Key findings

**Track A** (English/EU ratio):
1. **More EU data helps**: A3 (30% EU) = 58.8% > A2 (20%) = 57.2% > A1 (10%) = 54.8% on m-arena-hard-EU.
2. **Severe English regression**: All models drop to ~13% winrate on arena-hard (~-37pp). The degradation is roughly constant regardless of EU ratio, suggesting continued training itself causes forgetting.
3. **Elo (balanced, w/ en)**: With language-balanced battles, A3 (709) approaches the baseline (766). A1 (613) and A2 (600) remain below — English regression still hurts on the English portion of battles.
4. **Transfer**: Romanian (60-65%) transfers well despite not being in training data. Greek (50-51%, different script) does not.
5. **Czech** performs best (74%) — likely underrepresented in the baseline.
6. **French barely moves** (46-53%).

**Track B** (data scaling):
7. **More data does not help**: B1 (53.6%) and B2 (52.9%) score below their Track A counterparts A1 (54.8%) and A2 (57.2%) on m-arena-hard-EU, despite ~5× more data.
8. **English regression persists**: B1 (13.0%) and B2 (14.2%) show the same ~37pp English drop as Track A.

**Track C** (English-only control):
9. **English regresses without EU data**: C0-100en (no EU data at all) drops English to 11.8% on arena-hard.
10. **C0 EU winrate below parity**: 48.9% overall on m-arena-hard-EU.

**Track D** (Dolci English replay):
11. **Dolci replay preserves English**: D1 (54.6%), D2 (54.6%), D3 (54.3%) all maintain English arena-hard winrate across all EU ratios.
12. **D2-80en Elo (777±41 Q3, 757±8 Q3.5)** exceeds or matches the baseline.
13. **EU winrates**: D3 (63.5%) ≈ D1 (63.4%) > D2 (62.0%) on m-arena-hard-EU. Per-language, D3 leads on de (75.8%), es (72.0%), fr (63.9%), it (63.0%).
14. **D-track w/o English Elo (190-219)** is lower than A-track (235-261). Per-language analysis confirms: D-track's advantage is **entirely from English** (26pp gap) and **Romanian** (15pp gap). On the 8 trained EU languages, A and D perform within ±5pp — Dolci replay preserves English but doesn't improve multilingual transfer beyond what A-track achieves.
15. **Greek**: D1 (62.0%) does not replicate in D2 (51.2%) or D3 (51.5%).

**Track E** (Dolci replay at scale):
16. **E1-90en achieves the highest Elo** (808±51 Q3, 760±7 Q3.5). English: 57.0% on arena-hard.
17. **E1 w/o English Elo (258±29) is the highest** across all models — E-track scales better for multilingual when removing the English advantage.
18. **E2/E3 m-arena-hard-EU drop**: E2 (56.6%) and E3 (53.1%) are below E1 and their Track D counterparts. French drops sharply: E2 41.2%, E3 39.2% (vs E1 61.3%).
19. **Q3.5 judge tightens CIs dramatically**: Qwen3.5-27B produces ±7-11 CIs vs Qwen3-30B-A3B's ±41-142. Rankings remain consistent (D/E > A/B > C) but the variance gap between models is much smaller.

### Code

- Training configs: [`oellm/configs/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/configs)
- Evaluation scripts: [`oellm/evaluations/benchmarks/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/evaluations/benchmarks)
