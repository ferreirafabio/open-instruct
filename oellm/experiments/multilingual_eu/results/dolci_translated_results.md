# Dolci-Translated SFT Results

## Motivation

Previous multilingual experiments (Tracks A-G in `multilingual_eu`) used machine-translated data from multiple sources (fusion-synth, WildChat, lmsys, oasst2). This experiment uses Elaine's high-quality translations of the original Dolci-Instruct-SFT data (`openeurollm/Dolci-Instruct-SFT-translated`), covering 7 EU languages.

## Setup

| | |
|---|---|
| **Base checkpoint** | dolci-instruct-sft-v2-horeka (reproduced OLMo-3-7B-Instruct-SFT) |
| **English source** | allenai/Dolci-Instruct-SFT (2,152,112 samples) |
| **Translated source** | openeurollm/Dolci-Instruct-SFT-translated (495k/lang) |
| **Training languages** | en, cs, de, es, fi, fr, it, sv (7 translated) |
| **Eval languages** | en, cs, de, es, fi, fr, it, sv (same 8 languages) |
| **Judge** | Qwen3.5-27B (LMArena, 200 battles/lang, 100 bootstraps) |
| **Training** | 2 epochs, LR=8e-5, batch=1M tokens, seq_len=32768 |
| **Checkpointing** | ephemeral=100, permanent=500 |

## Experiment Tracks

| Track | Question | Design |
|-------|----------|--------|
| A | Does En/EU ratio matter with high-quality translations? | 75/25% and 25/75% en, Dolci replay. Training: en, cs, de, es, fi, fr, it, sv. Eval: same 8 languages via LMArena Elo. |

## Experiment Matrix

| Exp. | En/EU | English samples | Translated samples | Per-lang samples | Total samples | Elo | Elo en | Elo w/o en |
|------|-------|-----------|-------------|------------|---------|-----|--------|------------|
| **A-75en** | 75/25 | 2,152,112 | 717,370 | 102,481 | 2,869,482 | tba | tba | tba |
| **A-25en** | 25/75 | 1,155,000 | 3,465,000 | 495,000 | 4,620,000 | tba | 930±22 | tba |
| **Baseline** | 100/0 | 2,152,112 | 0 | 0 | 2,152,112 | tba | 950±21 | tba |

## Per-language Elo (Qwen3.5-27B, 200 battles/lang)

| Exp. | en | cs | de | es | fi | fr | it | sv |
|------|----|----|----|----|----|----|----|----|
| **A-75en** | tba | tba | tba | tba | tba | tba | tba | tba |
| **A-25en** | 930±22 | 764±26 | 711±33 | 753±32 | 796±33 | 753±27 | 808±25 | 769±32 |
| **Baseline** | 950±21 | 696±29 | 647±40 | 691±37 | 681±48 | 675±32 | 771±24 | 757±35 |

Note: fi (96 LMArena entries) and sv (87 entries) have fewer battles available, expect wider CIs.

## Key Findings (preliminary, A-25en only)

1. **English preserved**: A-25en English Elo (930±22) is within CI of baseline (950±21). Dolci replay works.
2. **All 7 translated languages improve**: Czech +68, German +64, Spanish +62, Finnish +115, French +78, Italian +37, Swedish +12.
3. **Finnish shows the largest gain** (+115 Elo), suggesting it benefits most from dedicated training data.
4. **Italian gains are modest** (+37), possibly because the baseline already performs well on Italian (771).

## Code

- Configs: `oellm/configs/dolci_translated_A_{75en,25en}.yaml`
- Training: `oellm/experiments/multilingual_eu/scripts/dolci_translated/train_{kislurm,horeka,horeka_h100}.sh`
- Evaluation: `oellm/experiments/multilingual_eu/scripts/dolci_translated/run_elo_per_language.sh`
- Tests: `oellm/tests/test_dolci_translated.py`
