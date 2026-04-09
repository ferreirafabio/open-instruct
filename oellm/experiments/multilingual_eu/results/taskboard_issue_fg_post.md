## Update: Track F — extreme En/EU ratios with Dolci replay

Following up on the Track A–E results above, we mapped the full English/EU ratio curve (100% → 0% English) with Dolci replay.

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

### Per-language Elo (all tracks)

![Per-language heatmap](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_per_language_heatmap.png?raw=true)

### Ratio curve

![Ratio curve](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_ratio_curve.png?raw=true)

### English vs EU tradeoff (all tracks)

![Tradeoff scatter](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_tradeoff_scatter.png?raw=true)

### Observations

1. **English Elo scales with English ratio**: F1 (954) → F2 (947) → F3 (935) → F4 (912) → F5 (762). Dolci replay preserves English well up to ~50% EU, then degrades. At 0% English, it collapses to ~760.
2. **75/25 is the overall sweet spot**: F2 and G2 achieve the highest overall Elo (747, 751) and highest EU WR% (51.4%, 60.2%) in their respective tracks.
3. **Pure EU training hurts EU performance too**: F5 (0% English) EU WR = 36.6%, below baseline (50%). Without English, the model loses general instruction-following ability.
4. **G-track confirms F-track pattern with matched samples**: G2 (75/25, 166k) achieves 751 Elo — comparable to E1 (758, 491k at 90/10). The ratio matters more than the total sample count.
5. **Per-language: non-English Elo is relatively flat across ratios**. The main lever is English preservation. EU languages don't clearly benefit from more EU training data in the mix.

Full results with all tracks: [multilingual_eu_results.md](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/multilingual_eu_results.md#full-experiment-matrix)
