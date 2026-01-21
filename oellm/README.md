# oellm: OLMo-3 SFT Experiments

This directory contains tools and workflows for training and evaluating OLMo-3 7B models with supervised fine-tuning (SFT).

## Directory Structure

```
oellm/
├── README.md                # This file
├── configs/                 # Language mixing configurations
│   ├── mixture_eu24_100en.yaml      # 100% English baseline
│   ├── mixture_eu24_90en_10eu.yaml  # 90% EN, 10% EU languages
│   └── mixture_eu24_80en_20eu.yaml  # 80% EN, 20% EU languages
├── pipelines/               # Data processing pipelines
│   ├── preprocessing/       # Download and prepare English datasets
│   ├── translation/         # Translate to EU languages
│   └── tokenization/        # Tokenize for training
├── utils/                   # Shared utilities
│   ├── checkpoint.py        # Atomic checkpointing with resume
│   ├── language_mixer.py    # Per-language sampling
│   ├── chunking.py          # Memory-efficient chunked output
│   └── convert_to_hf.sh     # Convert checkpoints to HuggingFace format
├── train/                   # Training scripts
├── evaluations/             # Model evaluation tools
├── experiments/             # Experiment-specific configs
│   └── baseline_repro/      # DOLCi baseline reproduction
└── tests/                   # Unit and integration tests
```

## Overview

| Directory | Purpose |
|-----------|---------|
| `pipelines/` | Data processing: preprocessing → translation → tokenization |
| `configs/` | Language mixing configurations for EU24 experiments |
| `utils/` | Shared utilities (checkpointing, mixing, chunking) |
| `train/` | SLURM training scripts |
| `evaluations/` | Benchmarks (AlpacaEval, ArenaHard) and manual testing |
| `experiments/` | Experiment-specific data preparation |
| `tests/` | Unit and integration tests (113 passing) |

---

## Training Workflows

### Baseline Reproduction (100% English)

Reproduce the official OLMo-3 7B DOLCi baseline training.

```bash
# 1. Download and preprocess English datasets
sbatch oellm/pipelines/preprocessing/download_datasets.sh

# 2. Tokenize for training
sbatch oellm/pipelines/tokenization/tokenize_baseline.sh

# 3. Launch training
DATASET_PATH=data/datasets_mixture_sft_tokenized \
sbatch oellm/train/train_think_sft_dolci_7b_slurm.sh
```

### EU24 Multilingual (Translation + Language Mixing)

Train on translated data with configurable language ratios.

```bash
# 1. Download and preprocess English datasets (if not done)
sbatch oellm/pipelines/preprocessing/download_datasets.sh

# 2. Download translation model
sbatch oellm/pipelines/translation/setup/download_model_600m.sh

# 3. Translate to EU languages
sbatch oellm/pipelines/translation/translate_slurm.sh

# 4. Tokenize with language mixing (e.g., 90% EN, 10% EU)
CONFIG=mixture_eu24_90en_10eu.yaml sbatch oellm/pipelines/tokenization/tokenize_eu24.sh

# 5. Launch training
DATASET_PATH=data/datasets_eu24_tokenized/eu24_90en_10eu \
sbatch oellm/train/train_think_sft_dolci_7b_slurm.sh
```

### Available Language Mixing Configs

| Config | English | EU Languages |
|--------|---------|--------------|
| `mixture_eu24_100en.yaml` | 100% | 0% |
| `mixture_eu24_90en_10eu.yaml` | 90% | 10% |
| `mixture_eu24_80en_20eu.yaml` | 80% | 20% |

---

## Evaluation

### Download Baselines

```bash
# Instruct baseline
sbatch oellm/evaluations/download_instruct_baseline.sh

# Think baseline
sbatch oellm/evaluations/download_think_baseline.sh
```

### Benchmark Evaluation

```bash
# Evaluate on all benchmarks
sbatch oellm/evaluations/benchmarks/run_evaluation.sh instruct all

# Evaluate on specific benchmark
sbatch oellm/evaluations/benchmarks/run_evaluation.sh think alpaca-eval
```

### Manual Testing

```bash
sbatch oellm/evaluations/manual/serve_model.sh instruct
# Check logs for Gradio URL
```

---

## Convert Checkpoints to HuggingFace Format

```bash
sbatch oellm/utils/convert_to_hf.sh instruct  # or: think
```

---

## Running Tests

```bash
# Run all tests (113 passing)
uv run pytest oellm/tests/ -v

# Run specific test file
uv run pytest oellm/tests/test_checkpoint.py -v
```

---

## Environment Setup

Ensure you have the `.env` file configured:

```bash
# .env
PROJECT_ROOT=/work/dlclarge2/ferreira-oellm/open-instruct
HF_TOKEN=<your-huggingface-token>
```

All SLURM scripts load this automatically.
