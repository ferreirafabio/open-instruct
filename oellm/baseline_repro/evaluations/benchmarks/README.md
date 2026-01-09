# Benchmark Evaluation

Automated LLM-judge evaluation using [OpenJury](https://github.com/OpenEuroLLM/OpenJury).

## Overview

This folder contains tools for automated benchmark evaluation using LLM judges. The evaluation:
- Compares trained models against the official `allenai/Olmo-3-7B-Instruct-SFT` baseline
- Uses a strong judge model (e.g., Qwen2.5-32B) to evaluate response quality
- Runs on standard benchmarks: AlpacaEval, Arena-Hard

## Setup (One-Time)

Run the setup script on the cluster:

```bash
cd /work/dlclarge2/ferreira-oellm/open-instruct/oellm/baseline_repro/evaluations/benchmarks
bash setup.sh
```

This will:
1. Clone OpenJury
2. Install langchain dependencies
3. Pre-download evaluation datasets to `data/openjury-eval-data/`

## Running Evaluations

### Quick Start

```bash
cd /work/dlclarge2/ferreira-oellm/open-instruct

# Run with defaults (alpaca-eval, 100 instructions)
sbatch oellm/baseline_repro/evaluations/benchmarks/run_evaluation.sh

# Specify dataset
sbatch oellm/baseline_repro/evaluations/benchmarks/run_evaluation.sh alpaca-eval

# Specify dataset and number of instructions
sbatch oellm/baseline_repro/evaluations/benchmarks/run_evaluation.sh arena-hard 500
```

### Check Results

```bash
# Watch the log
tail -f oellm/baseline_repro/logs/eval_*.log
```

Output shows winrates:

```
============================================================
                  🏆 MODEL BATTLE RESULTS 🏆                  
📊 Dataset: alpaca-eval
🤖 Competitors: Model A: baseline vs Model B: trained
⚖️ Judge: Qwen2.5-32B-Instruct
📈 Results Summary:
   Total Battles: 100
   Win Rate (A): 45.0%
   ✅ Wins:   45
   ❌ Losses: 50
   🤝 Ties:   5
============================================================
```

### Customize Evaluation

Edit `run_evaluation.sh` to change:
- `BASELINE` - Reference model path
- `TRAINED` - Your model path  
- `JUDGE_MODEL` - Judge model (bigger = better quality)
- `DATASET` - alpaca-eval, arena-hard, m-arena-hard-EU

## Files

| File | Description |
|------|-------------|
| `run_evaluation.sh` | SLURM script to run evaluation |
| `setup.sh` | One-time setup script |
| `analyse_results.py` | Analyze winrates from completed evaluations |
| `OpenJury/` | Cloned evaluation framework (after setup) |

## Benchmarks

| Benchmark | Description | Instructions |
|-----------|-------------|--------------|
| alpaca-eval | General instruction following | ~800 |
| arena-hard | Challenging multi-turn tasks | ~500 |
| m-arena-hard-EU | Multilingual (European) | ~500 |

## Judge Models

Choose based on your needs:

| Model | Quality | Speed | GPU Memory |
|-------|---------|-------|------------|
| `Qwen3-30B-A3B-Instruct-2507` | High | Medium | ~40GB |
| `Meta-Llama-3.1-70B-Instruct-FP8` | Very High | Slow | ~80GB |
| `Qwen2.5-32B-Instruct-GPTQ-Int8` | Medium | Fast | ~20GB |

## Troubleshooting

### Job Failed
Check logs:
```bash
cat oellm/baseline_repro/logs/eval_<jobid>.log
```

### Model Not Found
Ensure model paths in `run_evaluation.sh` are correct:
```bash
BASELINE="$PROJECT_ROOT/models/baselines/Olmo-3-7B-Instruct-SFT"
TRAINED="$PROJECT_ROOT/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf"
```

### OpenJury Not Found
Run setup first:
```bash
bash setup.sh
```

