## Update: Track F — extreme En/EU ratios with Dolci replay

Following up on the Track A–E results above, we mapped the full English/EU ratio curve (100% → 0% English) with Dolci replay.

### Design

100/75/50/25/0% English, ~500k target, 6 EU langs (de,es,fr,it,pt,pl), Dolci replay. Sample counts not fully matched due to Polish (7k) and Italian (29k) data caps.

### Results

| Exp. | En/EU | N | Elo | Elo en | Elo w/o en | EU WR% | en WR% |
|---|---|---|---|---|---|---|---|
| **F1-100en** | 100/0 | 500k | 728±9 | **954±22** | 695±11 | 52.2% | **57.6%** |
| **F2-75en** | 75/25 | 486k | **747±10** | **947±22** | 711±10 | 51.4% | **59.7%** |
| **F3-50en** | 50/50 | 453k | 726±9 | **935±21** | 700±12 | 44.7% | **55.9%** |
| **F4-25en** | 25/75 | 359k | 739±9 | **912±22** | 685±11 | 43.3% | **57.2%** |
| **F5-0en** | 0/100 | 234k | 677±11 | 762±31 | 680±11 | 36.6% | 9.7% |

Elo: Qwen3.5-27B judge, LMArena Bradley-Terry, 200 battles/lang. EU/en WR%: Qwen3-30B-A3B judge, vs baseline.

### Per-language Elo (all tracks)

![Per-language heatmap](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_per_language_heatmap.png?raw=true)

### Ratio curve

![Ratio curve](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_ratio_curve.png?raw=true)

### English vs EU tradeoff (all tracks)

![Tradeoff scatter](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/fg_tradeoff_scatter.png?raw=true)

### Observations

1. **English Elo scales with English ratio**: F1 (954) > F2 (947) > F3 (935) > F4 (912) > F5 (762). Dolci replay preserves English well up to ~50% EU, then degrades. At 0% English, collapses to ~760.
2. **75/25 is the overall sweet spot**: F2 achieves the highest overall Elo (747) and competitive EU WR% (51.4%).
3. **Pure EU training hurts EU performance too**: F5 (0% English) EU WR = 36.6%, below baseline (50%). Without English, the model loses general instruction following ability.
4. **Non English Elo is relatively flat across ratios**. The main lever is English preservation. EU languages don't clearly benefit from more EU training data in the mix.

Full results with all tracks: [multilingual_eu_results.md](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/multilingual_eu_results.md#full-experiment-matrix)
