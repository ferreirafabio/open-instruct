---
license: apache-2.0
base_model: allenai/Olmo-3-7B-Instruct-SFT
language:
- en
library_name: transformers
tags:
- olmo
- sft
- checkpoints
- training-curve
---

# OLMo-3-7B-Instruct-SFT Training Checkpoints

This repository contains 4 intermediate checkpoints from a supervised fine-tuning (SFT) run of [OLMo-3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) on the [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-7B) dataset. These checkpoints are intended for studying how model performance evolves over the course of SFT training.

Following the OLMo 3 paper (Section 5.2.2), instruct SFT is warm-started from the think SFT checkpoint ([OLMo-3-7B-Think-SFT step42856](https://huggingface.co/openeurollm/OLMo-3-7B-Think-SFT)), not from the base model.

## Checkpoints

Checkpoints are stored in subdirectories named `step{N}/`.

| Step | Gap from prev |
|------|--------------|
| 1000 | - |
| 2000 | 1000 |
| 3000 | 1000 |
| 3252 | 252 |

Total training: 3,252 steps (~3.4B tokens at 1M tokens/step batch size, 2 epochs).

Training follows the hyperparameters reported in Table 47 (Section A.6.1) of the [OLMo 3 paper](https://arxiv.org/pdf/2512.13961):

| | 7B Instruct SFT |
|---|---|
| Total Tokens | ~3.4B |
| Learning Rate | 8.0 x 10⁻⁵ |
| Batch Size | 1M tokens |
| Max Sequence Length | 32K |
| Epochs | 2 |
| Packing | Yes |

## Usage

Each checkpoint is a standalone HuggingFace model. Load a specific checkpoint:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

step = 3252
model = AutoModelForCausalLM.from_pretrained(
    "openeurollm/OLMo-3-7B-Instruct-SFT",
    subfolder=f"step{step}",
)
tokenizer = AutoTokenizer.from_pretrained(
    "openeurollm/OLMo-3-7B-Instruct-SFT",
    subfolder=f"step{step}",
)
```

## Training Details

- **Base model**: [allenai/Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B)
- **Warm-start from**: [OLMo-3-7B-Think-SFT](https://huggingface.co/openeurollm/OLMo-3-7B-Think-SFT) (step42856, final)
- **Training data**: [allenai/Dolci-Instruct-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-7B)
- **Tokenizer**: [allenai/Olmo-3-7B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT) (includes function-calling chat template)
- **Precision**: bfloat16
- **Framework**: OLMo-core (converted to HuggingFace format)

## License

Apache 2.0
