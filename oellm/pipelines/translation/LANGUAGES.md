# Dataset Language Analysis

Analysis of source languages in the SFT dataset mixture.

## Summary

| Dataset | Rows | EN% | Primary Languages |
|---------|------|-----|-------------------|
| Nemotron | 6.3M | 100% | — |
| Smoltalk | 2.1M | 84% | fr:11, es:5, de:5 |
| PerfectBlend | 1.4M | 98% | ml:1, it:1, da:1 |
| Orca-Agent | 1.0M | 100% | — |
| LMSYS-1M | 1.0M | 76% | pt:10, es:7, ru:6 |
| SHP | 385,563 | 100% | — |
| Nectar | 182,954 | 98% | af:1, da:1, ca:1 |
| HH-RLHF | 169,352 | 98% | af:1, es:1 |
| UltraFeedback | 60,917 | 100% | fr:1 |
| Arena-55k | 57,477 | 96% | it:3, fr:3, af:1 |
| Comparia (FR) | 25,542 | 3% | fr:190, es:2, da:1 |
| Orca-DPO | 12,859 | 100% | — |
| Aegis | 12,628 | 98% | pt:1 |
| HumanLLMs | 10,884 | 99% | hr:1, af:1 |
| HelpSteer2 | 9,125 | 99% | fr:1, pt:1 |
| Capybara | 7,563 | 100% | — |
| Math-DPO | 2,418 | 100% | — |
| MT-Bench | 1,282 | 100% | — |
| Truthy | 1,016 | 100% | — |

## Language Codes

| Code | Language |
|------|----------|
| en | English |
| fr | French |
| de | German |
| es | Spanish |
| pt | Portuguese |
| ru | Russian |
| zh-cn | Chinese (Simplified) |
| ja | Japanese |
| ko | Korean |
| ? | Undetected (code/short text) |

## Key Findings

### Datasets to EXCLUDE from EN→X translation:
- **Comparia (FR)**: 100% French - translate FROM French, not to it
- **Orca-Agent**: Mostly code/structured data, not natural language

### Multilingual datasets (need source language detection):
- **LMSYS-1M**: ~32% non-English (ru, zh-cn, es, pt, de, ja, ko, etc.)
- **Arena-55k**: ~10% non-English

### Mostly English (safe for EN→X translation):
- All other datasets: 90%+ English

## Methodology

- Sampled 200 rows per dataset
- Used `langdetect` library (character n-gram classifier, not neural)
- Checked first non-empty message with 30+ characters
- `?` indicates undetectable (code, very short, or special characters)