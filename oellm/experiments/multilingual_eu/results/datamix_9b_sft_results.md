# Datamix-9B Instruct SFT Results

## Motivation

The OpenEuroLLM 9B alpha model ([datamix-9b-80-20](https://huggingface.co/openeurollm/datamix-9b-80-20)) is a LlamaForCausalLM pretrained on 4B tokens with an 80/20 English/multilingual datamix. We run instruct SFT using [Dolci-Instruct-SFT translated with gemma-3-27b-it](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-SFT-translated) data to evaluate where the model stands after post-training.

## Setup

| | |
|---|---|
| **Base model** | openeurollm/datamix-9b-80-20 (9B, LlamaForCausalLM, Gemma tokenizer) |
| **English source** | allenai/Dolci-Instruct-SFT (2,152,112 samples) |
| **Translated source** | openeurollm/Dolci-Instruct-SFT-translated (495k/lang) |
| **Data mix** | 75% English / 25% translated (dt-A-75en, 2.87M samples) |
| **Training languages** | en, cs, de, es, fi, fr, it, sv |
| **Eval languages** | en, cs, de, es, fi, fr, it, sv |
| **Judge** | Qwen3.5-27B (LMArena, 500 battles/lang, 100 bootstraps) |
| **Training** | 2 epochs, LR=8e-5, batch=128, seq_len=2048, DDP (no DeepSpeed) |
| **Chat template** | simple_chat (model has no built-in template) |

## Per-language Elo (Qwen3.5-27B, 500 battles/lang)

| Model | en | cs | de | es | fi | fr | it | sv |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|
| **datamix-9b-sft** | 879±16 | **833±15** | **792±19** | **764±20** | tba | **790±18** | 780±18 | **820±32** |
| **OLMo-3-7B A-75en** | **950±14** | 714±19 | 690±24 | 746±18 | 732±44 | 743±17 | **820±15** | 722±35 |
| **OLMo-3-7B Baseline** | 954±16 | 647±23 | 632±27 | 745±20 | 688±48 | 695±23 | 766±17 | 777±33 |

![Per-language Elo comparison](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/datamix_9b_per_language_elo.png?raw=true)

## Key Findings (preliminary, fi pending)

1. **datamix-9b strongly outperforms OLMo on non-English languages**: Czech +119, German +102, Swedish +98, French +47 over OLMo A-75en.
2. **English is weaker** (879±16 vs OLMo A-75en 950±14), likely due to shorter context (2048 vs 32768) and the base model being an early alpha.
3. **Czech is the standout** (833±15), highest non-English Elo across all models. The 80/20 multilingual pretraining datamix gives a strong multilingual foundation.
4. **Swedish transfers well** (820±32) despite the base model's 80/20 datamix focusing on higher-resource languages.

## Code

- Training: `oellm/experiments/datamix_9b_sft/scripts/train_kislurm.sh`
- Evaluation: per-language Elo via LMArena
- Tests: `oellm/tests/test_datamix_9b_sft.py`
