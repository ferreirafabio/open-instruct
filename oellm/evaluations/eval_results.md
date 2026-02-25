# Experiment Log Book

Evaluation results collected across experiments.

<details><summary>Helper code (click to expand)</summary>

```python
import json
from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown

RESULTS_ROOT = Path("oellm/evaluations/benchmarks/OpenJury/results")

def load_winrate(results_dir):
    rows = []
    for f in sorted((RESULTS_ROOT / results_dir).rglob("results-*.json")):
        d = json.loads(f.read_text())
        if d.get("eval_mode") != "winrate": continue
        rows.append({"Benchmark": d["dataset"], "Baseline WR%": d["winrate"]*100,
            "Ours WR%": (1-d["winrate"])*100, "Battles": d["num_battles"]})
    return pd.DataFrame(rows)

def load_rubric(results_dir):
    rows = []
    for f in sorted((RESULTS_ROOT / results_dir).rglob("results-*.json")):
        d = json.loads(f.read_text())
        if d.get("eval_mode") != "rubric": continue
        for c in d["criteria"]:
            rows.append({"Benchmark": d["dataset"], "Criterion": c,
                "Baseline": d["model_A_scores"][f"{c}_score"],
                "Ours": d["model_B_scores"][f"{c}_score"]})
    return pd.DataFrame(rows)
```

</details>

---

# Reproducing OLMo-3-7B-SFT

## Think SFT v2 (HoreKa) — Winrate

| | |
|---|---|
| **Ours** | `checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-horeka-hf` |
| **Baseline** | `models/baselines/Olmo-3-7B-Think-SFT` |
| **Judge** | `Qwen/Qwen3-30B-A3B-Instruct-2507` (winrate mode, both orderings) |
| **Results** | `oellm/evaluations/benchmarks/OpenJury/results/horeka-winrate-Olmo-3-7B-Think-SFT-20260224_141545` |
| **Date** | 2026-02-24 |

### Winrate summary

Baseline is fixed at 50.0% as reference. Values > 50% = better than baseline. *(from `analyse_results.py`)*

<table style="font-size:1.1em">
<thead><tr><th>Model</th><th>alpaca-eval</th><th>arena-hard</th><th>m-arena-hard-EU</th><th>Average</th></tr></thead>
<tbody>
<tr><td>baselines/Olmo-3-7B-Think-SFT</td><td>0.500</td><td>0.500</td><td>0.500</td><td>0.500</td></tr>
<tr style="font-weight:bold"><td>olmo3-7b-sft/dolci-think-sft-v2-horeka-hf</td><td>0.511</td><td>0.505</td><td>0.512</td><td>0.509</td></tr>
</tbody></table>

### Detailed battle counts

Head-to-head comparison with per-benchmark win/loss/tie counts. *(from `summarize_preferences.py`)*

<table style="font-size:1.1em">
<thead><tr><th>Benchmark</th><th>Baseline WR%</th><th>Ours WR%</th><th>Battles</th><th>Wins</th><th>Losses</th><th>Ties</th></tr></thead>
<tbody>
<tr><td>alpaca-eval</td><td>48.9</td><td>51.1</td><td>1610</td><td>819</td><td>783</td><td>8</td></tr>
<tr><td>arena-hard</td><td>49.5</td><td>50.5</td><td>1000</td><td>503</td><td>493</td><td>4</td></tr>
<tr><td>m-arena-hard-EU</td><td>48.8</td><td>51.2</td><td>12000</td><td>6129</td><td>5838</td><td>33</td></tr>
<tr style="font-weight:bold"><td>Average</td><td>49.1</td><td>50.9</td><td>14610</td><td>7451</td><td>7114</td><td>45</td></tr>
</tbody></table>

<details><summary>Code</summary>

```python
show_winrate(
    title="Think SFT v2 (HoreKa) — Winrate",
    results_dir="horeka-winrate-Olmo-3-7B-Think-SFT-20260224_141545",
    ours_path="checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-horeka-hf",
    baseline_path="models/baselines/Olmo-3-7B-Think-SFT",
    judge="Qwen/Qwen3-30B-A3B-Instruct-2507 (winrate, both orderings)",
    date="2026-02-24",
)
```

</details>

---

*Add new experiments below following the same pattern.*
