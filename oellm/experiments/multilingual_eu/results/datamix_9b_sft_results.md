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
| **datamix-9b-sft** | 879±16 | **833±15** | **792±19** | **764±20** | **815±35** | **790±18** | 780±18 | **820±32** |
| **OLMo-3-7B-Instruct-SFT A-75en** | **950±14** | 714±19 | 690±24 | 746±18 | 732±44 | 743±17 | **820±15** | 722±35 |
| **OLMo-3-7B-Instruct-SFT** | 954±16 | 647±23 | 632±27 | 745±20 | 688±48 | 695±23 | 766±17 | 777±33 |

![Per-language Elo comparison](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/datamix_9b_per_language_elo.png?raw=true)

## Key Findings

1. **datamix-9b strongly outperforms OLMo on non-English languages**: Czech +119, German +102, Swedish +98, Finnish +83, French +47 over OLMo-3-7B-Instruct-SFT A-75en.
2. **English is weaker** (879±16 vs OLMo-3-7B-Instruct-SFT A-75en 950±14), likely due to shorter context (2048 vs 32768) and the base model being an early alpha.
3. **Czech is the standout** (833±15), highest non-English Elo across all models. The 80/20 multilingual pretraining datamix gives a strong multilingual foundation.
4. **Swedish transfers well** (820±32) despite the base model's 80/20 datamix focusing on higher-resource languages.
5. **Finnish strongly improved** (815±35 vs OLMo baseline 688±48, +127), showing the multilingual pretraining datamix benefits even lower-resource languages.

## 2048 Token Truncation for All Models

For a fair comparison, we reran both OLMo models with the same 1024/1024 token limit used for datamix-9b (LMArena, 500 battles/language, 100 bootstraps):

| Model | en | cs | de | es | fi | fr | it | sv |
|---|---|---|---|---|---|---|---|---|
| **datamix-9b-sft** | 879±16 | **833±15** | **792±19** | 764±20 | **815±35** | 790±18 | 780±18 | **820±32** |
| **OLMo-3-7B-Instruct-SFT A-75en (1024/1024)** | 970±14 | 733±18 | 742±20 | **820±17** | 752±38 | **804±16** | **821±15** | 786±32 |
| **OLMo-3-7B-Instruct-SFT (1024/1024)** | **976±15** | 690±19 | 722±22 | 799±18 | 812±36 | 756±20 | 766±15 | 805±37 |

**Impact on the datamix-9b vs OLMo A-75en comparison (both at 1024/1024):**
- cs: datamix 833 vs OLMo 733 → +100 (was +119)
- de: datamix 792 vs OLMo 742 → +50 (was +102)
- es: datamix 764 vs OLMo 820 → -56 (datamix now losing, was +18)
- fi: datamix 815 vs OLMo 752 → +63 (was +83)
- fr: datamix 790 vs OLMo 804 → -14 (datamix now losing, was +47)
- it: datamix 780 vs OLMo 821 → -41
- sv: datamix 820 vs OLMo 786 → +34 (was +98)

datamix-9b still beats OLMo A-75en on cs, de, fi, sv, but now loses on es, fr, it.

**Note:** The Qwen3.5-27B judge appears to prefer shorter, more direct answers. When OLMo models are truncated to 1024/1024, they produce shorter outputs that align better with the judge's preferences, explaining the performance improvement in English and some non-English languages.

![Per-language Elo comparison (1024/1024 truncation)](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/datamix_9b_per_language_elo_truncated.png?raw=true)

## Code

- Training: `oellm/experiments/datamix_9b_sft/scripts/train_kislurm.sh`
- Evaluation: per-language Elo via LMArena
- Tests: `oellm/tests/test_datamix_9b_sft.py`
