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
| **Judge (Elo)** | Qwen3.5-27B (dense, via vllm serve, both orderings, 8k max tokens) |
| **Judge (winrate)** | Qwen3-30B-A3B-Instruct-2507 (VLLM, both orderings, 8k max tokens) |
| **Elo benchmark** | LMArena, Bradley-Terry, 100 bootstraps, 200 battles/language, 12 EU languages |
| **Winrate benchmarks** | m-arena-hard-EU (6000 prompts, 12 EU languages), arena-hard (500 English prompts) |
| **Training languages** | en + de, es, fr, it, pt, pl, nl, cs (ratio varies by experiment) |
| **Eval languages** | en, de, es, fr, it, pt, pl, nl, cs, ro, el, uk (12 languages available in LMArena) |
| **Held-out languages** | ro, el (zero-shot transfer test, not in training data) |

### Experiment tracks

| Track | Question | Design |
|---|---|---|
| **A** | Preliminary: does the English/EU ratio matter? | 90/80/70% English, ~95k samples, fusion-synth only, EU share split equally across 8 languages |
| **B** | Does more diverse data help? | Same ratios, ~490k samples, fusion-synth + wildchat, lmsys-chat, oasst2 |
| **C** | Is English regression caused by EU data, or by continued SFT itself? | 100% English control (no EU data), same total samples |
| **D** | Does replaying the base checkpoint's English data reduce forgetting? | Same ratios as Track A, but English from Dolci-Instruct-SFT (base checkpoint's own training data) |
| **E** | Does Dolci replay scale with more data? | Same as Track D, but ~490k samples instead of ~95k |

### Full experiment matrix

| Exp. | En/EU | N | Elo | Elo en | Elo w/o en | EU WR% | en WR% |
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

Elo: Qwen3.5-27B judge. "en" = English-only (200 battles). "w/o en" = 11 non-English languages.

EU WR% / en WR% = winrate vs baseline (Qwen3-30B-A3B judge). 50% = parity.

### Findings

1. **Dolci replay preserves English** (Track D): English Elo stays at 942-963 (baseline is 950), English arena-hard winrate ~55%. Without replay (Track A), English drops to 766-771 Elo / ~13% winrate. The source of the English data matters, the En/EU ratio doesn't.
2. **The D-track advantage is almost entirely English**: per-language Elo shows A vs D gap is ~190 points on English. On trained EU languages, they're within CIs.
3. **Scaling helps modestly**: E1 (90/10 at 491k) reaches the highest overall Elo (758). More EU data at scale (E2/E3) shows diminishing returns.
4. **More diverse data doesn't help** (Track B): 5× more data from wildchat/lmsys/oasst2 → similar Elo to Track A.
5. **Continued SFT itself causes forgetting** (Track C): C0 (100% English, no EU data at all) still drops English Elo to 791. The regression is not caused by EU data but by training on non-Dolci English.
6. **Transfer**: Romanian (held-out, Latin script) transfers well (Elo 700-890). Greek (held-out, different script) does not (540-690).

### Code

- Training configs: [`oellm/configs/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/configs)
- Evaluation scripts: [`oellm/evaluations/benchmarks/`](https://github.com/ferreirafabio/open-instruct/tree/main/oellm/evaluations/benchmarks)
- Full results: [`multilingual_eu_results.md`](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/multilingual_eu_results.md)
