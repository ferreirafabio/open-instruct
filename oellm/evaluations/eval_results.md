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

## OLMo-3-7B Think SFT: Winrate

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
    title="OLMo-3-7B Think SFT (HoreKa): Winrate",
    results_dir="horeka-winrate-Olmo-3-7B-Think-SFT-20260224_141545",
    ours_path="checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-horeka-hf",
    baseline_path="models/baselines/Olmo-3-7B-Think-SFT",
    judge="Qwen/Qwen3-30B-A3B-Instruct-2507 (winrate, both orderings)",
    date="2026-02-24",
)
```

</details>

## OLMo-3-7B Think SFT: Rubric

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
    title="OLMo-3-7B Think SFT (HoreKa): Rubric",
    results_dir="horeka-rubric-Olmo-3-7B-Think-SFT-20260224_161039",
    ours_path="checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-horeka-hf",
    baseline_path="models/baselines/Olmo-3-7B-Think-SFT",
    judge="Qwen/Qwen3-30B-A3B-Instruct-2507 (rubric)",
    date="2026-02-24",
)
```

</details>

### ELO Ratings (LMArena)

Bradley-Terry ELO estimated on 20k LMArena battles (all languages, single-turn), 100 bootstraps. Judged by Qwen3-30B-A3B-Instruct-2507 (swap_mode=both, 8k max tokens).

| Model | ELO (LMArena) |
|---|---|
| Think-SFT (official) | 1002.9 ± 9.0 |
| Think-SFT (ours, step 42856) | 1002.2 ± 13.0 |

---

## OLMo-3-7B Think SFT Performance over Training Time: Reproduction vs. Baseline

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

![Training curve evaluation overview](https://raw.githubusercontent.com/ferreirafabio/open-instruct/main/oellm/evaluations/figures/training_curve_eval.png)

![Per-criterion rubric scores](https://raw.githubusercontent.com/ferreirafabio/open-instruct/main/oellm/evaluations/figures/rubric_criteria_eval.png)

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
| 13000 | 0.936 | 0.889 | -0.047 | 0.859 | 0.808 | -0.051 |
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

### ELO (LMArena, 20k battles)

Bradley-Terry ELO estimated on 20k LMArena battles (all languages, single-turn), 100 bootstraps. Same prompts across all checkpoints (deterministic sampling, seed=0).

![Think ELO training curve](https://raw.githubusercontent.com/ferreirafabio/open-instruct/main/oellm/evaluations/figures/think_elo_curve.png)

| Step | ELO (LMArena) |
|---:|---|
| 500 | 980.4 ± 11.8 |
| 1000 | 941.2 ± 7.7 |
| 2000 | 916.5 ± 9.3 |
| 3000 | 907.7 ± 9.2 |
| 4000 | 937.2 ± 10.1 |
| 5000 | 915.2 ± 9.1 |
| 7000 | 947.0 ± 8.0 |
| 8000 | 915.4 ± 10.6 |
| 11000 | 932.5 ± 7.0 |
| 13000 | 949.2 ± 7.7 |
| 15000 | 943.3 ± 9.7 |
| 17000 | 928.2 ± 8.3 |
| 19000 | 961.6 ± 7.0 |
| 21000 | 955.0 ± 7.1 |
| 24000 | 979.8 ± 8.2 |
| 27000 | 973.4 ± 7.6 |
| 31000 | 1012.4 ± 5.6 |
| 34000 | 1000.5 ± 12.5 |
| 38000 | 991.4 ± 8.7 |
| 42856 | 1002.2 ± 13.0 |

## OLMo-3-7B Instruct SFT: Winrate

| | |
|---|---|
| **Ours** | `checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka-hf` |
| **Baseline** | `models/baselines/Olmo-3-7B-Instruct-SFT` |
| **Judge** | `Qwen/Qwen3-30B-A3B-Instruct-2507` (winrate mode, both orderings) |
| **Results** | `oellm/evaluations/benchmarks/OpenJury/results/instruct-v2-horeka-Olmo-3-7B-Instruct-SFT-20260228_192129` |
| **Date** | 2026-02-28 |

### Winrate summary

Baseline is fixed at 50.0% as reference. Values > 50% = better than baseline.

| Benchmark | Baseline | Ours | Delta |
|---|---|---|---|
| alpaca-eval | 0.500 | 0.488 | -0.012 |
| arena-hard | 0.500 | 0.528 | +0.028 |
| m-arena-hard-EU | 0.500 | 0.527 | +0.027 |
| | | | |
| **Average** | **0.500** | **0.514** | **+0.014** |

### Detailed battle counts

Head-to-head comparison with per-benchmark win/loss/tie counts.

| Benchmark | Baseline WR% | Ours WR% | Delta | Battles | Wins | Losses | Ties |
|---|---|---|---|---|---|---|---|
| alpaca-eval | 51.2 | 48.8 | -2.4 | 1610 | 756 | 794 | 60 |
| arena-hard | 47.3 | 52.8 | +5.5 | 1000 | 520 | 465 | 15 |
| m-arena-hard-EU | 47.0 | 52.3 | +5.3 | 12000 | 6281 | 5641 | 78 |
| | | | | | | | |
| **Average** | **48.5** | **51.3** | **+2.8** | **14610** | **7557** | **6900** | **153** |

<details><summary>Code</summary>

```python
show_winrate(
    title="OLMo-3-7B Instruct SFT (HoreKa): Winrate",
    results_dir="instruct-v2-horeka-Olmo-3-7B-Instruct-SFT-20260228_192129",
    ours_path="checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka-hf",
    baseline_path="models/baselines/Olmo-3-7B-Instruct-SFT",
    judge="Qwen/Qwen3-30B-A3B-Instruct-2507 (winrate, both orderings)",
    date="2026-02-28",
)
```

</details>

## OLMo-3-7B Instruct SFT: Rubric

| | |
|---|---|
| **Ours** | `checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka-hf` |
| **Baseline** | `models/baselines/Olmo-3-7B-Instruct-SFT` |
| **Judge** | `Qwen/Qwen3-30B-A3B-Instruct-2507` (rubric mode) |
| **Results** | `oellm/evaluations/benchmarks/OpenJury/results/instruct-v2-horeka-Olmo-3-7B-Instruct-SFT-20260227_205048` |
| **Date** | 2026-02-28 |

### alpaca-eval

| Criterion | Baseline | Ours | Delta |
|---|---|---|---|
| Instruction Following | 0.594 | 0.594 | +0.001 |
| Naturalness | 0.693 | 0.697 | +0.004 |
| Coherence | 0.652 | 0.657 | +0.006 |
| Accuracy | 0.589 | 0.589 | 0.000 |
| | | | |
| **Average ¹** | **0.887** | **0.891** | **+0.004** |

### arena-hard

| Criterion | Baseline | Ours | Delta |
|---|---|---|---|
| Instruction Following | 0.540 | 0.550 | +0.009 |
| Naturalness | 0.692 | 0.690 | -0.001 |
| Coherence | 0.616 | 0.632 | +0.016 |
| Accuracy | 0.509 | 0.512 | +0.004 |
| | | | |
| **Average ¹** | **0.815** | **0.827** | **+0.012** |

### m-arena-hard-EU

| Criterion | Baseline | Ours | Delta |
|---|---|---|---|
| Instruction Following | 0.305 | 0.318 | +0.013 |
| Naturalness | 0.572 | 0.587 | +0.015 |
| Coherence | 0.413 | 0.425 | +0.012 |
| Accuracy | 0.280 | 0.293 | +0.013 |
| | | | |
| **Average ¹** | **0.488** | **0.510** | **+0.022** |

<details><summary>Code</summary>

```python
show_rubric(
    title="OLMo-3-7B Instruct SFT (HoreKa): Rubric",
    results_dir="instruct-v2-horeka-Olmo-3-7B-Instruct-SFT-20260227_205048",
    ours_path="checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka-hf",
    baseline_path="models/baselines/Olmo-3-7B-Instruct-SFT",
    judge="Qwen/Qwen3-30B-A3B-Instruct-2507 (rubric)",
    date="2026-02-28",
)
```

</details>

### ELO Ratings (LMArena)

Bradley-Terry ELO estimated on 20k LMArena battles (all languages, single-turn), 100 bootstraps. Judged by Qwen3-30B-A3B-Instruct-2507 (swap_mode=both, 8k max tokens).

| Model | ELO (LMArena) |
|---|---|
| Instruct-SFT (official) | 940.5 ± 7.9 |
| Instruct-SFT (ours, step 3252) | 953.0 ± 9.3 |

---

## OLMo-3-7B Instruct SFT Performance over Training Time: Reproduction vs. Baseline

| | |
|---|---|
| **Checkpoints** | `checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka-hf-all/step{N}` (4 steps: 1000–3252) |
| **Baseline** | `models/baselines/Olmo-3-7B-Instruct-SFT` |
| **Judge** | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| **Benchmarks** | alpaca-eval (805 prompts), arena-hard (500 prompts), m-arena-hard-EU (6000 prompts) |
| **Eval modes** | winrate (fixed + both orderings), rubric (1-7 Likert, 4 criteria) |
| **Results** | `oellm/evaluations/benchmarks/OpenJury/results/instruct-v2-curve-*` |
| **Data** | `oellm/experiments/instruct_v2_checkpoint_eval/results.csv` |
| **Date** | 2026-02-28 |

### Overview plots

![Training curve evaluation overview](https://raw.githubusercontent.com/ferreirafabio/open-instruct/main/oellm/evaluations/figures/instruct_training_curve_eval.png)

![Per-criterion rubric scores](https://raw.githubusercontent.com/ferreirafabio/open-instruct/main/oellm/evaluations/figures/instruct_rubric_criteria_eval.png)

### Winrate (swap_mode=both)

Our model win% vs baseline. 50% = parity. Both orderings (A-B and B-A) averaged. Battles per step: 1610 (alpaca-eval), 1000 (arena-hard), 12000 (m-arena-hard-EU).

| Step | alpaca-eval | arena-hard | m-arena-hard-EU |
|---:|---:|---:|---:|
| 1000 | 42.6% | 35.3% | 35.8% |
| 2000 | 47.1% | 50.3% | 48.4% |
| 3000 | 47.5% | 55.5% | 52.3% |
| 3252 | 48.8% | 52.8% | 54.5% |

### Winrate (swap_mode=fixed)

Single ordering only (baseline=A, ours=B). Battles per step: 805 (alpaca-eval), 500 (arena-hard), 6000 (m-arena-hard-EU).

| Step | alpaca-eval | arena-hard | m-arena-hard-EU |
|---:|---:|---:|---:|
| 1000 | 37.8% | 33.9% | 38.3% |
| 2000 | 46.2% | 42.5% | 50.9% |
| 3000 | 45.8% | 46.3% | 55.4% |
| 3252 | 45.2% | 46.2% | 54.9% |

### Rubric: Average ¹ scores

Average score (0–1 scale, higher = better). Delta = Ours - Baseline.

| Step | alpaca-eval Baseline | alpaca-eval Ours | Delta | arena-hard Baseline | arena-hard Ours | Delta | m-arena-hard-EU Baseline | m-arena-hard-EU Ours | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.887 | 0.857 | -0.030 | 0.815 | 0.755 | -0.060 | 0.489 | 0.419 | -0.071 |
| 2000 | 0.886 | 0.884 | -0.002 | 0.821 | 0.815 | -0.005 | 0.487 | 0.488 | +0.001 |
| 3000 | 0.891 | 0.888 | -0.003 | 0.826 | 0.840 | +0.014 | 0.487 | 0.511 | +0.024 |
| 3252 | 0.887 | 0.891 | +0.004 | 0.815 | 0.827 | +0.012 | 0.489 | 0.510 | +0.021 |

### ELO (LMArena, 20k battles)

Bradley-Terry ELO estimated on 20k LMArena battles (all languages, single-turn), 100 bootstraps. Same prompts across all checkpoints (deterministic sampling, seed=0).

![Instruct ELO training curve](https://raw.githubusercontent.com/ferreirafabio/open-instruct/main/oellm/evaluations/figures/instruct_elo_curve.png)

| Step | ELO (LMArena) |
|---:|---|
| 1000 | 889.0 ± 7.6 |
| 2000 | 945.9 ± 10.0 |
| 3000 | 954.2 ± 9.2 |
| 3252 | 953.0 ± 9.3 |

---

<sup>1</sup> Average (0–1) is a min-max normalization of the mean of the 4 criterion scores.

---

## Training Throughput

Throughput metrics for the v2 production runs (used for all evaluations above). Think SFT v2 started on kislurm and was transferred to HoreKa to finish; Instruct SFT v2 ran entirely on HoreKa. All metrics from wandb offline logs.

**Metric definitions:**
- **TPS (Tokens Per Second)**: Number of tokens one GPU processes per second. Total cluster throughput = TPS/device × 8 GPUs.
- **MFU (Model FLOPs Utilization)**: Percentage of the GPU's theoretical peak compute used for training (forward + backward FLOPs per the PaLM convention, corrected for sliding-window attention). See [MFU Calculation](#mfu-calculation) below for the full derivation.
- **Wall-clock**: Total training compute time = `total_steps × avg_step_time`. Excludes SLURM queue waits and restart overhead.

### Hardware & Configuration

| | |
|---|---|
| **GPUs** | 8× NVIDIA H200 SXM (141 GiB HBM3e) |
| **Topology** | Think: 1×8 H200 (kislurm, steps 0–35K) → 2×4 H200 (HoreKa, steps 35K–42.8K). Instruct: 2×4 H200 (HoreKa) |
| **Intra-node** | NVLink 4th gen (900 GB/s bidirectional) |
| **Inter-node** | InfiniBand HDR200 (200 Gb/s), Mellanox ConnectX HCA (`mlx5`) — HoreKa only |
| **Precision** | BF16 |
| **Seq length** | 32,768 |
| **Global batch size** | 1,048,576 tokens (1M) |
| **Tokens/step/device** | 131,072 |

### Throughput Summary

| Metric | Think SFT | Instruct SFT |
|---|---|---|
| Total tokens | 44.9B | 3.4B |
| Total steps | 42,856 | 3,252 |
| Epochs | 2 | 2 |
| Dataset size | 694,030 samples | 54,314 samples |
| Learning rate | 5e-5 | 8e-5 |
| | | |
| **TPS/device (avg)** | **7,704** | **8,731** |
| **Total TPS (8 GPUs)** | **61,634** | **69,846** |
| Step time (avg) | 17.0 s | 15.0 s |
| Steps/hour | 212 | 240 |
| Tokens/GPU-hour | 27.7M | 31.4M |
| | | |
| **MFU (H200 BF16 dense)** | **46.0%** | **52.1%** |
| Data loading overhead | 0.15% | 0.15% |
| | | |
| GPU active memory | 110.5 GiB (79.0%) | 110.5 GiB (79.0%) |
| GPU reserved memory | 118.6 GiB (84.8%) | 118.6 GiB (84.8%) |
| | | |
| **Computed wall-clock** | **202.5 h** | **13.6 h** |
| **GPU-hours** | **1,620** | **108** |

### Per-Step Observations

- **Think SFT**: TPS/device stable at 7,500–8,000 across 42K steps. Low outliers (~2,500–4,500 TPS) at session restarts are warmup/compilation overhead after SLURM preemptions.
- **Instruct SFT**: TPS/device stable at 8,500–9,100 across full training. Higher throughput than Think due to higher masked-label ratio (56.1% vs 3.3%), reducing effective computation per token.
- **Data loading**: Negligible overhead (0.15% of step time, ~0.2 ms per step).
- **Memory**: Identical footprint for both models — expected since model architecture and batch size are identical.

### Throughput Plots

![TPS over training steps](https://raw.githubusercontent.com/ferreirafabio/open-instruct/main/oellm/evaluations/figures/throughput_tps.png?v=3)

![MFU over training steps](https://raw.githubusercontent.com/ferreirafabio/open-instruct/main/oellm/evaluations/figures/throughput_mfu.png?v=3)

### MFU Calculation

Following the [PaLM convention](https://arxiv.org/abs/2204.02311) (Appendix B). OLMo-3-7B uses a `[sliding, sliding, sliding, full]` repeating pattern = **24 sliding-window layers (w=4,096) + 8 full-attention layers** (from `config.json` `layer_types`). The attention FLOPs are computed per-layer using each layer's actual attention length:

```
MFU = flops_per_token × TPS_per_device / peak_FLOPS_per_device

flops_per_token = 6 × N_non_embedding + attention_flops

N_non_embedding   = 6,887,575,552
peak_FLOPS        = 989.5 × 10¹²                                       (H200 SXM BF16 dense: 1,979 / 2)

Dense matmul term:
  6 × 6,887,575,552 = 41,325,453,312

Attention term (12 × n_layers × n_heads × head_dim × attn_length):
  Full-attention layers (8):   12 × 8  × 32 × 128 × 32,768 = 12,884,901,888
  Sliding-window layers (24):  12 × 24 × 32 × 128 × 4,096  =  4,831,838,208
  Total attention:                                           = 17,716,740,096

Total flops_per_token = 41,325,453,312 + 17,716,740,096 = 59,042,193,408

Think MFU    = 59,042,193,408 × 7,704  / 989.5 × 10¹² = 46.0%
Instruct MFU = 59,042,193,408 × 8,731  / 989.5 × 10¹² = 52.1%
```

Plotting code: [`plot_throughput.py`](../experiments/baseline_repro/scripts/plot_throughput.py).

