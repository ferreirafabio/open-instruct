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
| **G** | Same as F but with matched sample counts (removing data size confound) | 100/75/50/25/0% English, 166k samples (matched), 4 EU languages (de,es,fr,pt — drop pl/it bottleneck), Dolci replay |

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
| **G1-100en** | 100/0 | 166k | 739±9 | **987±22** | 702±11 | 53.6% | **57.3%** |
| **G2-75en** | 75/25 | 166k | **751±9** | **951±21** | 709±10 | **60.2%** | **57.4%** |
| **G3-50en** | 50/50 | 166k | 737±9 | **914±21** | 708±11 | 55.0% | **56.6%** |
| **G4-25en** | 25/75 | 166k | 722±9 | **961±23** | 694±11 | 47.4% | **53.7%** |
| **G5-0en** | 0/100 | 166k | tba | tba | tba | tba | tba |

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

### Per-language Elo ratings (Q3.5 judge, LMArena)

Bradley-Terry Elo computed independently per language (200 battles/language, Qwen3.5-27B judge). Higher = better.

| Model | en | de | es | fr | it | pt | pl | nl | cs | ro | el | uk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | 950±21 | 647±40 | 691±37 | 675±32 | 771±24 | 704±37 | 721±30 | 749±29 | 696±29 | 863±45 | 641±74 | 664±34 |
| **A1-90en** | 771±32 | 613±40 | 750±34 | 612±42 | 771±28 | 650±46 | 634±40 | 714±27 | 718±29 | 726±60 | 661±57 | 677±36 |
| **A2-80en** | 769±30 | 574±46 | 735±37 | 663±39 | 779±26 | 671±37 | 732±33 | 712±32 | 675±34 | 688±79 | 640±70 | 666±39 |
| **A3-70en** | 766±29 | 515±58 | 733±38 | 693±35 | 784±26 | 706±38 | 721±34 | 706±35 | 684±34 | 845±51 | 618±68 | 634±37 |
| **B1-90en** | 789±26 | 615±43 | 692±32 | 675±33 | 776±26 | 705±37 | 742±30 | 717±34 | 732±27 | 875±47 | 595±79 | 620±40 |
| **B2-80en** | 797±26 | 668±37 | 698±34 | 711±33 | 807±24 | 729±37 | 740±31 | 690±32 | 676±38 | 815±54 | 662±56 | 652±38 |
| **C0-100en** | 791±29 | 612±41 | 670±38 | 654±41 | 756±27 | 641±35 | 691±36 | 689±29 | 648±33 | 705±58 | 686±63 | 650±33 |
| **D1-90en** | 942±20 | 619±39 | 744±34 | 672±36 | 766±26 | 651±42 | 712±35 | 718±35 | 697±32 | 888±51 | 644±66 | 687±35 |
| **D2-80en** | 956±20 | 589±43 | 708±32 | 699±36 | 746±28 | 683±42 | 663±38 | 742±32 | 706±29 | 854±58 | 688±52 | 656±41 |
| **D3-70en** | 962±20 | 643±40 | 708±32 | 663±34 | 784±26 | 706±37 | 734±34 | 723±35 | 710±33 | 845±55 | 627±77 | 664±36 |
| **E1-90en** | 965±21 | 642±40 | 711±34 | 695±36 | 807±25 | 712±31 | 732±34 | 758±27 | 703±37 | 754±63 | 604±65 | 634±37 |
| **E2-80en** | 931±24 | 681±36 | 726±34 | 608±46 | 751±31 | 645±38 | 705±36 | 754±29 | 706±30 | 838±52 | 673±59 | 634±40 |
| **E3-70en** | 940±22 | 613±44 | 721±36 | 682±32 | 781±27 | 676±43 | 676±34 | 757±27 | 694±30 | 844±49 | 658±67 | 642±38 |
| **F1-100en** | 954±22 | 614±39 | 718±32 | 664±42 | 742±28 | 680±40 | 634±38 | 714±33 | 677±33 | 892±49 | 636±66 | 671±37 |
| **F2-75en** | 947±22 | 624±44 | 713±36 | 702±34 | 776±30 | 697±36 | 708±32 | 708±32 | 666±34 | 854±49 | 612±74 | 624±37 |
| **F3-50en** | 935±20 | 658±44 | 696±35 | 712±30 | 788±27 | 635±55 | 750±30 | 679±33 | 653±34 | 825±50 | 652±75 | 637±38 |
| **F4-25en** | 912±22 | 601±40 | 732±32 | 646±45 | 800±24 | 638±43 | 715±29 | 649±36 | 661±35 | 784±52 | 605±66 | 689±32 |
| **F5-0en** | 762±30 | 607±46 | 685±36 | 619±41 | 775±26 | 601±50 | 686±34 | 641±39 | 621±39 | 718±69 | 575±87 | 564±46 |
| **G1-100en** | 987±22 | 549±45 | 733±33 | 685±34 | 781±27 | 700±28 | 680±35 | 740±26 | 586±37 | 819±53 | 596±73 | 623±39 |
| **G2-75en** | 950±21 | 685±34 | 696±35 | 689±35 | 757±28 | — | 711±32 | 781±29 | 665±32 | 878±52 | 676±62 | 675±33 |
| **G3-50en** | 914±21 | 631±42 | 715±35 | 707±34 | 788±26 | 710±38 | 667±33 | 649±38 | 647±35 | 841±50 | 659±63 | 655±31 |
| **G4-25en** | 960±23 | 623±46 | 712±35 | 676±34 | 774±27 | 651±40 | 656±37 | 682±30 | 661±31 | 826±54 | 660±60 | 656±37 |
| **G5-0en** | 776±24 | 640±40 | 701±33 | 696±31 | 751±28 | 628±40 | 636±33 | 646±35 | 585±46 | 833±49 | 543±86 | 650±37 |

### Per-language Elo heatmap

![Per-language Elo heatmap](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_per_language_heatmap.png?raw=true)

### English vs EU tradeoff (all tracks)

![English vs EU tradeoff](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_tradeoff_scatter.png?raw=true)

### Elo vs English/EU ratio (Track F)

![Elo vs ratio](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_ratio_curve.png?raw=true)

### Key findings

1. **Dolci replay preserves English** (Track D): English Elo stays at 942-963 (baseline is 950), English arena-hard winrate ~55%. Without replay (Track A), English drops to 766-771 Elo / ~13% winrate. The source of the English data matters, the En/EU ratio doesn't.
2. **The D-track advantage is almost entirely English**: per-language Elo shows A vs D gap is ~190 points on English. On trained EU languages, they're within CIs.
3. **Scaling helps modestly**: E1 (90/10 at 491k) reaches the highest overall Elo (758). More EU data at scale (E2/E3) shows diminishing returns.
4. **More diverse data doesn't help** (Track B): 5x more data from wildchat/lmsys/oasst2, similar Elo to Track A.
5. **Continued SFT itself causes forgetting** (Track C): C0 (100% English, no EU data at all) still drops English Elo to 791. The regression is not from the EU data, it's from training on non-Dolci English.
6. **75/25 is the sweet spot** (Track F): F2 achieves the highest overall Elo (747). English Elo degrades gracefully up to ~50% EU, then collapses at 0% English (762). Pure EU training (F5) hurts EU performance too (below baseline).
7. **Transfer**: Romanian (held-out, Latin script) transfers well (Elo 700-890). Greek (held-out, different script) does not (540-690).

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
