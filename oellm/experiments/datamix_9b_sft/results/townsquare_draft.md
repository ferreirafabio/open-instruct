Hi everyone,

We ran instruct SFT on the OpenEuroLLM 9B alpha model ([datamix-9b-80-20](https://huggingface.co/openeurollm/datamix-9b-80-20)) using [Dolci-Instruct-SFT translated with gemma-3-27b-it](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-SFT-translated) (translated by Elaine). Per-language Elo ratings below (LMArena, 500 battles/language, 100 bootstraps):

[barplot image]

| Model | en | cs | de | es | fi | fr | it | sv |
|---|---|---|---|---|---|---|---|---|
| **datamix-9b-sft** | 879±16 | **833±15** | **792±19** | **764±20** | **815±35** | **790±18** | 780±18 | **820±32** |
| **OLMo-3-7B-Instruct-SFT A-75en** | **950±14** | 714±19 | 690±24 | 746±18 | 732±44 | 743±17 | **820±15** | 722±35 |
| **OLMo-3-7B-Instruct-SFT** | **954±16** | 647±23 | 632±27 | 745±20 | 688±48 | 695±23 | 766±17 | 777±33 |

### Setup

| | datamix-9b-sft | OLMo-3-7B-Instruct-SFT A-75en |
|---|---|---|
| **Base model** | openeurollm/datamix-9b-80-20 (9B, LlamaForCausalLM) | OLMo-3-7B-Instruct-SFT (7B, reproduced baseline) |
| **Data** | 75% en Dolci + 25% translated (2.87M samples) | 75% en Dolci + 25% translated (2.87M samples) |
| **Training languages** | en, cs, de, es, fi, fr, it, sv | en, cs, de, es, fi, fr, it, sv |
| **Max context** | 2048 | 32768 |
| **Training** | 2 epochs, LR=8e-5, batch=128 | 2 epochs, LR=8e-5, batch=1M tokens |

Note: datamix-9b has a 2048-token context limit. For evaluation, we split this into 1024 tokens for the input prompt and 1024 tokens for the model output. The OLMo models were evaluated with 8192 tokens for both input and output (32K context), so the comparison is not fully apples-to-apples -- datamix-9b is disadvantaged on longer prompts.

The model outperforms OLMo-3-7B-Instruct-SFT on most non-English languages (e.g. Czech +186, German +160, Finnish +127) while being behind in English (879 vs 954). These are preliminary results -- the base model was trained on only 1T tokens with an old datamix/hyperparameters and has not yet been context-length extended (2K vs 32K for OLMo).

Full results: https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/datamix_9b_sft_results.md
