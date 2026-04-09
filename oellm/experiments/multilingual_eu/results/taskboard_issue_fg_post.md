## Update: Track F & G — extreme En/EU ratio experiments

Following up on the Track A–E results above, we ran two additional tracks to map the full English/EU ratio curve with Dolci replay.

### New tracks

| Track | Question | Design |
|---|---|---|
| **F** | How do extreme En/EU ratios affect performance with Dolci replay? | 100/75/50/25/0% English, ~500k target, 6 EU langs (de,es,fr,it,pt,pl), Dolci replay |
| **G** | Same as F but with matched sample count (removing data-size confound) | 100/75/50/25/0% English, 166k matched, 4 EU langs (de,es,fr,pt), Dolci replay |

Note: F-track sample counts are not fully matched because Polish (7k available) and Italian (29k) cap the high-EU-ratio configs. G-track fixes this by using only 4 high-resource languages and targeting the largest matched count (166k).

### Results

| Exp. | En/EU | N | Elo | Elo en | Elo w/o en | EU WR% | en WR% |
|---|---|---|---|---|---|---|---|
| **F1-100en** | 100/0 | 500k | 728±9 | **954±22** | 695±11 | 52.2% | **57.6%** |
| **F2-75en** | 75/25 | 486k | **747±10** | **947±22** | 711±10 | 51.4% | **59.7%** |
| **F3-50en** | 50/50 | 453k | 726±9 | **935±21** | 700±12 | 44.7% | **55.9%** |
| **F4-25en** | 25/75 | 359k | 739±9 | **912±22** | 685±11 | 43.3% | **57.2%** |
| **F5-0en** | 0/100 | 234k | 677±11 | 762±31 | 680±11 | 36.6% | 9.7% |
| | | | | | | | |
| **G1-100en** | 100/0 | 166k | 739±9 | **987±22** | 702±11 | 53.6% | **57.3%** |
| **G2-75en** | 75/25 | 166k | **751±9** | **951±21** | 709±10 | **60.2%** | **57.4%** |
| **G3-50en** | 50/50 | 166k | 737±9 | **914±21** | 708±11 | 55.0% | **56.6%** |
| **G4-25en** | 25/75 | 166k | 722±9 | **961±23** | 694±11 | 47.4% | **53.7%** |
| **G5-0en** | 0/100 | 166k | tba | tba | tba | tba | tba |

Elo: Qwen3.5-27B judge, LMArena Bradley-Terry, 200 battles/lang. EU/en WR%: Qwen3-30B-A3B judge, vs baseline.

### Per-language Elo (Qwen3.5-27B, 200 battles/lang)

| Model | en | de | es | fr | it | pt | pl | nl | cs | ro | el | uk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
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

### Ratio curve (F vs G)

![Ratio curve](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_ratio_curve.png?raw=true)

### English vs EU tradeoff (all tracks)

![Tradeoff scatter](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_tradeoff_scatter.png?raw=true)

### Per-language Elo heatmap

![Per-language heatmap](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_per_language_heatmap.png?raw=true)

### Observations

1. **English Elo scales with English ratio**: F1 (954) → F2 (947) → F3 (935) → F4 (912) → F5 (762). Dolci replay preserves English well up to ~50% EU, then degrades. At 0% English, it collapses to ~760.
2. **75/25 is the overall sweet spot**: F2 and G2 achieve the highest overall Elo (747, 751) and highest EU WR% (51.4%, 60.2%) in their respective tracks.
3. **Pure EU training hurts EU performance too**: F5 (0% English) EU WR = 36.6%, below baseline (50%). Without English, the model loses general instruction-following ability.
4. **G-track confirms F-track pattern with matched samples**: G2 (75/25, 166k) achieves 751 Elo — comparable to E1 (758, 491k at 90/10). The ratio matters more than the total sample count.
5. **Per-language: non-English Elo is relatively flat across ratios**. The main lever is English preservation. EU languages don't clearly benefit from more EU training data in the mix.

Full results with all tracks: [multilingual_eu_results.md](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/multilingual_eu_results.md#full-experiment-matrix)
