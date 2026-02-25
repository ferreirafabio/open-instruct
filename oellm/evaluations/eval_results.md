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

| Model | alpaca-eval | arena-hard | m-arena-hard-EU | Average |
|---|---|---|---|---|
| baselines/Olmo-3-7B-Think-SFT | 0.500 | 0.500 | 0.500 | 0.500 |
| **olmo3-7b-sft/dolci-think-sft-v2-horeka-hf** | **0.511** | **0.505** | **0.512** | **0.509** |
| **Delta** | **+0.011** | **+0.005** | **+0.012** | **+0.009** |

### Detailed battle counts

Head-to-head comparison with per-benchmark win/loss/tie counts. *(from `summarize_preferences.py`)*

| Benchmark | Baseline WR% | Ours WR% | Delta | Battles | Wins | Losses | Ties |
|---|---|---|---|---|---|---|---|
| alpaca-eval | 48.9 | 51.1 | +2.2 | 1610 | 819 | 783 | 8 |
| arena-hard | 49.5 | 50.5 | +1.0 | 1000 | 503 | 493 | 4 |
| m-arena-hard-EU | 48.8 | 51.2 | +2.4 | 12000 | 6129 | 5838 | 33 |
| | | | | | | | |
| **Average** | **49.1** | **50.9** | **+1.9** | **14610** | **7451** | **7114** | **45** |

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

## Think SFT v2 (HoreKa) — Rubric

| | |
|---|---|
| **Ours** | `checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-horeka-hf` |
| **Baseline** | `models/baselines/Olmo-3-7B-Think-SFT` |
| **Judge** | `Qwen/Qwen3-30B-A3B-Instruct-2507` (rubric mode) |
| **Results** | `oellm/evaluations/benchmarks/OpenJury/results/horeka-rubric-Olmo-3-7B-Think-SFT-20260224_161039` |
| **Date** | 2026-02-24 |

### alpaca-eval

| Criterion | Baseline | Ours | Delta |
|---|---|---|---|
| Instruction Following | 0.636 | 0.641 | +0.005 |
| Naturalness | 0.681 | 0.677 | -0.005 |
| Coherence | 0.670 | 0.666 | -0.004 |
| Accuracy | 0.648 | 0.647 | -0.001 |
| | | | |
| **Average ¹** | **0.931** | **0.930** | **-0.002** |

### arena-hard

| Criterion | Baseline | Ours | Delta |
|---|---|---|---|
| Instruction Following | 0.580 | 0.582 | +0.003 |
| Naturalness | 0.652 | 0.650 | -0.002 |
| Coherence | 0.627 | 0.631 | +0.003 |
| Accuracy | 0.589 | 0.595 | +0.006 |
| | | | |
| **Average ¹** | **0.853** | **0.857** | **+0.004** |

### m-arena-hard-EU

| Criterion | Baseline | Ours | Delta |
|---|---|---|---|
| Instruction Following | 0.497 | 0.502 | +0.006 |
| Naturalness | 0.603 | 0.608 | +0.005 |
| Coherence | 0.562 | 0.569 | +0.008 |
| Accuracy | 0.506 | 0.513 | +0.008 |
| | | | |
| **Average ¹** | **0.736** | **0.747** | **+0.011** |

<details><summary>Code</summary>

```python
show_rubric(
    title="Think SFT v2 (HoreKa) — Rubric",
    results_dir="horeka-rubric-Olmo-3-7B-Think-SFT-20260224_161039",
    ours_path="checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-horeka-hf",
    baseline_path="models/baselines/Olmo-3-7B-Think-SFT",
    judge="Qwen/Qwen3-30B-A3B-Instruct-2507 (rubric)",
    date="2026-02-24",
)
```

</details>

---

*Add new experiments below following the same pattern.*

---

<sup>1</sup> Average (0–1) is a min-max normalization of the mean of the 4 criterion scores.
