# Manual Evaluation

Interactive chat-based evaluation using vLLM and Gradio UI.

## Overview

This folder contains tools for manually testing and comparing models through an interactive chat interface. This is useful for:
- Qualitative assessment of model responses
- Testing specific prompts and edge cases
- Side-by-side comparison of trained vs. baseline models

## Available Models

| Model | Type | Description |
|-------|------|-------------|
| `instruct` | Trained | OLMo-3 7B trained with DOLCi Instruct SFT |
| `think` | Trained | OLMo-3 7B trained with DOLCi Think SFT |
| `baseline` | Reference | Official `allenai/Olmo-3-7B-Instruct-SFT` |

## Quick Start

### 1. Download Baseline (First Time Only)

```bash
cd /work/dlclarge2/ferreira-oellm/open-instruct
sbatch oellm/baseline_repro/evaluations/download_baseline.sh
```

### 2. Serve a Model

```bash
# Serve trained instruct model
sbatch oellm/baseline_repro/evaluations/manual/serve_model.sh instruct

# Serve trained think model
sbatch oellm/baseline_repro/evaluations/manual/serve_model.sh think

# Serve baseline for comparison
sbatch oellm/baseline_repro/evaluations/manual/serve_model.sh baseline
```

### 3. Access the Chat UI

Check the log for connection instructions:

```bash
tail -f oellm/baseline_repro/logs/serve_*.log
```

The log will show an SSH command like:
```bash
ssh -J kis3bat1.rz.ki.privat -L 7860:localhost:7860 -L 8000:localhost:8000 <node>
```

Then open http://localhost:7860 in your browser.

## Files

| File | Description |
|------|-------------|
| `serve_model.sh` | SLURM script to serve model with vLLM + Gradio UI |
| `chat_ui.py` | Gradio-based browser chat interface |

## Manual Setup (Alternative)

If you prefer to run vLLM and the UI separately:

```bash
# Terminal 1: Start vLLM server
srun -p alldlc2_gpu-h200 --gpus=1 --time=4:00:00 \
  /work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf \
  --served-model-name dolci-instruct \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000

# Terminal 2: Start Chat UI (on same node or with port forward)
python oellm/baseline_repro/evaluations/manual/chat_ui.py \
  --api-base http://localhost:8000 \
  --model dolci-instruct \
  --share
```

## Tips for Manual Evaluation

1. **Compare Side-by-Side**: Run two models in separate browser tabs to compare responses
2. **Test Edge Cases**: Try prompts that test specific capabilities (math, coding, reasoning)
3. **Check Instruction Following**: Test complex multi-step instructions
4. **Evaluate Safety**: Test for harmful content refusal
5. **Document Findings**: Note interesting differences between trained and baseline models

