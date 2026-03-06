---
license: apache-2.0
base_model: allenai/Olmo-3-7B-Think-SFT
language:
- en
library_name: transformers
tags:
- olmo
- sft
- checkpoints
- training-curve
---

# OLMo-3-7B-Think-SFT-v2 Training Checkpoints

This repository contains 20 intermediate checkpoints from a supervised fine-tuning (SFT) run of [OLMo-3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) on the [Dolci-Think-SFT](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) dataset. These checkpoints are intended for studying how model performance evolves over the course of SFT training.

We verified successful baseline reproduction against the official [allenai/Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) using LLM-judge evaluations (winrate and rubric modes on alpaca-eval and arena-hard benchmarks). Detailed evaluation results and methodology are available in the [evaluation results](https://github.com/ferreirafabio/open-instruct/blob/main/oellm/evaluations/eval_results.md).

## Checkpoints

Checkpoints are stored in subdirectories named `step{N}/`. Spacing is dense early (where changes are most dramatic) and gradually sparser later.

| Step | Gap from prev |
|------|--------------|
| 500 | - |
| 1000 | 500 |
| 2000 | 1000 |
| 3000 | 1000 |
| 4000 | 1000 |
| 5000 | 1000 |
| 7000 | 2000 |
| 8000 | 1000 |
| 11000 | 3000 |
| 13000 | 2000 |
| 15000 | 2000 |
| 17000 | 2000 |
| 19000 | 2000 |
| 21000 | 2000 |
| 24000 | 3000 |
| 27000 | 3000 |
| 31000 | 4000 |
| 34000 | 3000 |
| 38000 | 4000 |
| 42856 | 4856 |

Total training: 42,856 steps (~45.4B tokens at 1M tokens/step batch size, 2 epochs).

Training follows the hyperparameters reported in Table 47 (Section A.6.1) of the [OLMo 3 paper](https://arxiv.org/pdf/2512.13961):

| | 7B Thinking SFT |
|---|---|
| Total Tokens | 45.4B |
| Learning Rate | 5.0 x 10⁻⁵ |
| Batch Size | 1M tokens |
| Max Sequence Length | 32K |
| Epochs | 2 |
| Packing | Yes |

## Usage

Each checkpoint is a standalone HuggingFace model. Load a specific checkpoint:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

step = 5000
model = AutoModelForCausalLM.from_pretrained(
    "openeurollm/OLMo-3-7B-Think-SFT-v2",
    subfolder=f"step{step}",
)
tokenizer = AutoTokenizer.from_pretrained(
    "openeurollm/OLMo-3-7B-Think-SFT-v2",
    subfolder=f"step{step}",
)
```

## Training Details

- **Base model**: [allenai/Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B)
- **Training data**: [allenai/Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B)
- **Tokenizer**: [allenai/Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) (includes `<think>` chat template)
- **Precision**: bfloat16
- **Framework**: OLMo-core (converted to HuggingFace format)

## License

Apache 2.0
