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

## Think SFT: Winrate

| | |
|---|---|
| **Ours** | `checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-horeka-hf` |
| **Baseline** | `models/baselines/Olmo-3-7B-Think-SFT` |
| **Judge** | `Qwen/Qwen3-30B-A3B-Instruct-2507` (winrate mode, both orderings) |
| **Results** | `oellm/evaluations/benchmarks/OpenJury/results/horeka-winrate-Olmo-3-7B-Think-SFT-20260224_141545` |
| **Date** | 2026-02-24 |

### Winrate summary

Baseline is fixed at 50.0% as reference. Values > 50% = better than baseline. *(from `analyse_results.py`)*

| Benchmark | Baseline | Ours | Delta |
|---|---|---|---|
| alpaca-eval | 0.500 | 0.511 | +0.011 |
| arena-hard | 0.500 | 0.505 | +0.005 |
| m-arena-hard-EU | 0.500 | 0.512 | +0.012 |
| | | | |
| **Average** | **0.500** | **0.509** | **+0.009** |

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
    title="Think SFT (HoreKa): Winrate",
    results_dir="horeka-winrate-Olmo-3-7B-Think-SFT-20260224_141545",
    ours_path="checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-horeka-hf",
    baseline_path="models/baselines/Olmo-3-7B-Think-SFT",
    judge="Qwen/Qwen3-30B-A3B-Instruct-2507 (winrate, both orderings)",
    date="2026-02-24",
)
```

</details>

## Think SFT: Rubric

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
    title="Think SFT (HoreKa): Rubric",
    results_dir="horeka-rubric-Olmo-3-7B-Think-SFT-20260224_161039",
    ours_path="checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-horeka-hf",
    baseline_path="models/baselines/Olmo-3-7B-Think-SFT",
    judge="Qwen/Qwen3-30B-A3B-Instruct-2507 (rubric)",
    date="2026-02-24",
)
```

</details>

---

## Think SFT: Training Curve Evaluation

| | |
|---|---|
| **Checkpoints** | `checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-hf/step{N}` (19 steps: 500–38000) + final step 42856 |
| **Baseline** | `models/baselines/Olmo-3-7B-Think-SFT` |
| **Judge** | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| **Benchmarks** | alpaca-eval (805 prompts), arena-hard (500 prompts) |
| **Eval modes** | winrate (fixed + both orderings), rubric (1-7 Likert, 4 criteria) |
| **Results** | `oellm/evaluations/benchmarks/OpenJury/results/think-v2-curve-*` |
| **Data** | `oellm/experiments/think_v2_checkpoint_eval/all_results.csv` |
| **Date** | 2026-02-26 |

### Overview plots

![Training curve evaluation overview](figures/training_curve_eval.png)

![Per-criterion rubric scores](figures/rubric_criteria_eval.png)

### Winrate (swap_mode=both)

Our model win% vs baseline. 50% = parity. Both orderings (A-B and B-A) averaged. Battles per step: 1610 (alpaca-eval), 1000 (arena-hard).

| Step | alpaca-eval | arena-hard |
|---:|---:|---:|
| 500 | 40.8% | 32.5% |
| 1000 | 36.6% | 31.1% |
| 2000 | 35.3% | 32.0% |
| 3000 | 38.7% | 31.3% |
| 4000 | 41.4% | 35.2% |
| 5000 | 38.2% | 35.3% |
| 7000 | 43.5% | 39.1% |
| 8000 | 38.0% | 34.7% |
| 11000 | 40.7% | 37.4% |
| 13000 | 41.9% | 36.1% |
| 15000 | 42.7% | 38.8% |
| 17000 | 40.8% | 38.8% |
| 19000 | 44.0% | 42.9% |
| 21000 | 43.9% | 41.1% |
| 24000 | 48.8% | 47.5% |
| 27000 | 48.9% | 44.6% |
| 31000 | 49.6% | 48.6% |
| 34000 | 49.0% | 51.4% |
| 38000 | 49.4% | 48.5% |
| 42856 | 51.1% | 50.5% |

### Winrate (swap_mode=fixed)

Single ordering only (baseline=A, ours=B). ~35-40pp gap vs swap_mode=both results. Battles per step: 805 (alpaca-eval), 500 (arena-hard).

| Step | alpaca-eval | arena-hard |
|---:|---:|---:|
| 500 | 9.9% | 13.4% |
| 1000 | 8.1% | 14.7% |
| 2000 | 8.6% | 13.2% |
| 3000 | 10.7% | 14.4% |
| 4000 | 9.6% | 16.9% |
| 5000 | 7.0% | 15.5% |
| 7000 | 12.0% | 15.7% |
| 8000 | 8.6% | 13.9% |
| 11000 | 9.4% | 15.0% |
| 13000 | 10.0% | 17.8% |
| 15000 | 10.1% | 17.9% |
| 17000 | 10.0% | 18.1% |
| 19000 | 10.5% | 20.6% |
| 21000 | 9.8% | 19.1% |
| 24000 | 12.8% | 21.1% |
| 27000 | 13.5% | 22.5% |
| 31000 | 14.5% | 25.0% |
| 34000 | 14.8% | 21.9% |
| 38000 | 15.4% | 26.8% |
| 42856 | 13.8% | 24.3% |

### Rubric: Average ¹ scores

Average score (0–1 scale, higher = better). Delta = Ours - Baseline.

| Step | alpaca-eval Baseline | alpaca-eval Ours | Delta | arena-hard Baseline | arena-hard Ours | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.935 | 0.906 | -0.029 | 0.861 | 0.778 | -0.082 |
| 1000 | 0.940 | 0.880 | -0.060 | 0.861 | 0.781 | -0.079 |
| 2000 | 0.935 | 0.872 | -0.063 | 0.861 | 0.766 | -0.095 |
| 3000 | 0.929 | 0.870 | -0.058 | 0.859 | 0.767 | -0.092 |
| 4000 | 0.927 | 0.888 | -0.040 | 0.861 | 0.791 | -0.069 |
| 5000 | 0.920 | 0.878 | -0.042 | 0.853 | 0.780 | -0.073 |
| 7000 | 0.932 | 0.896 | -0.036 | 0.859 | 0.798 | -0.061 |
| 8000 | 0.928 | 0.890 | -0.038 | 0.859 | 0.797 | -0.062 |
| 11000 | 0.936 | 0.906 | -0.029 | 0.861 | 0.826 | -0.035 |
| 13000 | 0.936 | 0.889 | -0.047 | 0.859 | 0.808 | -0.050 |
| 15000 | 0.932 | 0.897 | -0.034 | 0.861 | 0.823 | -0.038 |
| 17000 | 0.920 | 0.892 | -0.028 | 0.870 | 0.822 | -0.048 |
| 19000 | 0.920 | 0.896 | -0.024 | 0.859 | 0.841 | -0.018 |
| 21000 | 0.926 | 0.900 | -0.025 | 0.875 | 0.816 | -0.059 |
| 24000 | 0.932 | 0.919 | -0.013 | 0.861 | 0.861 | +0.001 |
| 27000 | 0.938 | 0.933 | -0.006 | 0.861 | 0.853 | -0.007 |
| 31000 | 0.927 | 0.934 | +0.008 | 0.874 | 0.865 | -0.009 |
| 34000 | 0.920 | 0.926 | +0.006 | 0.861 | 0.842 | -0.019 |
| 38000 | 0.926 | 0.931 | +0.006 | 0.861 | 0.862 | +0.001 |
| 42856 | 0.931 | 0.930 | -0.002 | 0.853 | 0.857 | +0.004 |

---

<sup>1</sup> Average (0–1) is a min-max normalization of the mean of the 4 criterion scores.
