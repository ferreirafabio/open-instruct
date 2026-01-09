# Model Evaluation

This directory contains tools for evaluating trained OLMo-3 7B models.

## Structure

```
evaluations/
├── README.md                 # This file
├── download_baseline.sh      # Download official baseline model
├── manual/                   # Interactive chat-based evaluation
│   ├── README.md
│   ├── chat_ui.py
│   └── serve_model.sh
└── benchmarks/               # Automated LLM-judge evaluation
    ├── README.md
    ├── launch_evaluation.py
    └── analyse_results.py
```

## Evaluation Types

| Type | Folder | Description |
|------|--------|-------------|
| **Manual** | `manual/` | Interactive chat testing via vLLM + Gradio UI |
| **Benchmarks** | `benchmarks/` | Automated LLM-judge evaluation on standard benchmarks |

## Quick Start

### 1. Download Baseline (Required for Both)

```bash
cd /work/dlclarge2/ferreira-oellm/open-instruct
sbatch oellm/baseline_repro/evaluations/download_baseline.sh
```

### 2. Manual Evaluation

Interactive testing through a chat interface:

```bash
sbatch oellm/baseline_repro/evaluations/manual/serve_model.sh instruct
# Check log for Gradio URL
```

See [manual/README.md](manual/README.md) for details.

### 3. Benchmark Evaluation

Automated comparison using LLM judges:

```bash
cd oellm/baseline_repro/evaluations/benchmarks
python launch_evaluation.py  # From local machine with slurmpilot
python analyse_results.py    # After jobs complete
```

See [benchmarks/README.md](benchmarks/README.md) for setup and usage.

## Models

| Model | Path | Type |
|-------|------|------|
| Trained Instruct | `checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf` | Your model |
| Trained Think | `checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf` | Your model |
| Baseline SFT | `models/baselines/Olmo-3-7B-Instruct-SFT` | `allenai/Olmo-3-7B-Instruct-SFT` |
| Base Model | `models/baselines/Olmo-3-1025-7B` | `allenai/Olmo-3-1025-7B` (pre-training) |

