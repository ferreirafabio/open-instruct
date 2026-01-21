# Model Evaluation

This directory contains tools for evaluating trained OLMo-3 7B models.

## Structure

```
evaluations/
├── README.md                     # This file
├── download_instruct_baseline.sh # Download instruct baseline
├── download_think_baseline.sh    # Download think baseline
├── download_think_baseline_32b.sh # Download 32B think baseline
├── manual/                       # Interactive chat-based evaluation
│   └── chat_ui.py
└── benchmarks/                   # Automated LLM-judge evaluation
    ├── run_evaluation.sh        # Main evaluation script
    ├── analyse_results.py       # Results analysis
    ├── summarize_preferences.py # Preference summary
    └── OpenJury/                # OpenJury evaluation framework
```

## Evaluation Types

| Type | Folder | Description |
|------|--------|-------------|
| **Manual** | `manual/` | Interactive chat testing via vLLM + Gradio UI |
| **Benchmarks** | `benchmarks/` | Automated LLM-judge evaluation (AlpacaEval, ArenaHard) |

## Quick Start

### 1. Download Baselines

```bash
cd /work/dlclarge2/ferreira-oellm/open-instruct

# For instruct model evaluation
sbatch oellm/evaluations/download_instruct_baseline.sh

# For think model evaluation
sbatch oellm/evaluations/download_think_baseline.sh
```

### 2. Benchmark Evaluation

Run automated LLM-judge comparison against baseline:

```bash
# Evaluate instruct model on all benchmarks
sbatch oellm/evaluations/benchmarks/run_evaluation.sh instruct all

# Evaluate think model on specific benchmark
sbatch oellm/evaluations/benchmarks/run_evaluation.sh think alpaca-eval

# Options: alpaca-eval, arena-hard, m-arena-hard-EU, all
```

Results are saved to `oellm/evaluations/benchmarks/OpenJury/results/`.

### 3. Manual Evaluation

Interactive testing through a chat interface:

```bash
sbatch oellm/evaluations/manual/serve_model.sh instruct
# Check logs for Gradio URL
tail -f oellm/evaluations/logs/serve_*.log
```

## Models

| Model | Path | Type |
|-------|------|------|
| Trained Instruct | `checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf` | Your model |
| Trained Think | `checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf` | Your model |
| Baseline Instruct | `models/baselines/Olmo-3-7B-Instruct-SFT` | `allenai/Olmo-3-7B-Instruct-SFT` |
| Baseline Think | `models/baselines/Olmo-3-7B-Think-SFT` | `allenai/Olmo-3-7B-Think-SFT` |
