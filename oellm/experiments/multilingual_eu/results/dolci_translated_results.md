# Dolci-Translated SFT Results

## Motivation

Previous multilingual experiments (Tracks A-F in `multilingual_eu`) used machine-translated data from multiple sources (fusion-synth, WildChat, lmsys, oasst2). This experiment uses [Dolci-Instruct-SFT translated with gemma-3-27b-it](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-SFT-translated) into 7 EU languages.

## Setup

| | |
|---|---|
| **Base checkpoint** | dolci-instruct-sft-v2-horeka (reproduced OLMo-3-7B-Instruct-SFT) |
| **English source** | allenai/Dolci-Instruct-SFT (2,152,112 samples) |
| **Translated source** | openeurollm/Dolci-Instruct-SFT-translated (495k/lang) |
| **Training languages** | en, cs, de, es, fi, fr, it, sv (7 translated) |
| **Eval languages** | en, cs, de, es, fi, fr, it, sv (same 8 languages) |
| **Judge** | Qwen3.5-27B (LMArena, 500 battles/lang, 100 bootstraps) |
| **Training** | 2 epochs, LR=8e-5, batch=1M tokens, seq_len=32768 |


## Experiment Tracks

| Track | Question | Design |
|-------|----------|--------|
| A | Does En/EU ratio matter with high-quality translations? | 75/25% and 25/75% en, Dolci replay. Training: en, cs, de, es, fi, fr, it, sv. Eval: same 8 languages via LMArena Elo. |

## Experiment Matrix

| Exp. | En/EU | English samples | Translated samples | Per-lang samples | Total samples | Elo† | Elo en† | Elo w/o en† |
|------|-------|-----------|-------------|------------|---------|-----|--------|------------|
| **A-75en** | 75/25 | 2,152,112 | 717,370 | 102,481 | 2,869,482 | **789±7** | **950±14** | 755±8 |
| **A-25en** | 25/75 | 1,155,000 | 3,465,000 | 495,000 | 4,620,000 | **782±6** | 913±16 | **742±8** |
| **Baseline** | 100/0 | 2,152,112 | 0 | 0 | 2,152,112 | 762±7 | **954±16** | 697±9 |

† Elo: LMArena Bradley-Terry, Qwen3.5-27B judge, 500 battles/lang, 100 bootstraps.

## Per-language Elo (Qwen3.5-27B, 500 battles/lang)

| Exp. | en | cs | de | es | fi | fr | it | sv |
|------|----|----|----|----|----|----|----|----|
| **A-75en** | **950±14** | 714±19 | 690±24 | 746±18 | 732±44 | 743±17 | **820±15** | 722±35 |
| **A-25en** | 913±16 | **745±15** | **701±23** | **763±18** | **813±33** | **756±17** | **801±15** | 756±33 |
| **Baseline** | **954±16** | 647±23 | 632±27 | 745±20 | 688±48 | 695±23 | 766±17 | **777±33** |

Note: fi (96 LMArena entries) and sv (87 entries) have fewer battles available.

![Per-language Elo comparison](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/experiments/multilingual_eu/results/plots/dolci_translated_per_language_elo.png?raw=true)

## Key Findings

1. **A-75en preserves English perfectly** (950±14 vs baseline 954±16, within CI) while improving non-English (755±8 vs 697±9).
2. **A-25en has strongest non-English gains** but slightly lower English (913±16). Overall Elo is comparable to A-75en (782±6 vs 789±7).
3. **A-75en has the best Italian** (820±15, +54 over baseline). A-25en is close (801±15).
4. **A-25en has the best Finnish** (813±33, +125 over baseline). A-75en gains less (732±44, +44).
5. **More translated data helps non-English** (A-25en beats A-75en on cs, de, es, fi, fr) at the cost of ~37 points English.
6. **Swedish is the exception**: baseline (777±33) outperforms both A-75en (722±35) and A-25en (756±33).
7. **Both experiments beat baseline overall**: A-75en 789±7, A-25en 782±6, baseline 762±7.

## Code

- Configs: `oellm/configs/dolci_translated_A_{75en,25en}.yaml`
- Training: `oellm/experiments/multilingual_eu/scripts/dolci_translated/train_{kislurm,horeka,horeka_h100}.sh`
- Evaluation: `oellm/experiments/multilingual_eu/scripts/dolci_translated/run_elo_per_language.sh`
- Tests: `oellm/tests/test_dolci_translated.py`
