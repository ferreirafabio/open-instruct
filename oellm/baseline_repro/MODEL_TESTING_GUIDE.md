# Testing Trained Models

Guide for testing the OLMo-3 7B SFT models trained with DOLCi data.

## Trained Models

| Model | Checkpoint | Description |
|-------|-----------|-------------|
| Instruct | `checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft/step3394/` | Direct instruction-following |
| Think | `checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft/step43376/` | Chain-of-thought reasoning with `<think>` tags |

## Evaluation

For comprehensive model evaluation, see the [evaluations/](evaluations/) directory:

| Type | Location | Description |
|------|----------|-------------|
| **Manual** | [`evaluations/manual/`](evaluations/manual/) | Interactive chat testing via vLLM + Gradio UI |
| **Benchmarks** | [`evaluations/benchmarks/`](evaluations/benchmarks/) | Automated LLM-judge evaluation against baseline |

### Quick Evaluation Start

```bash
# Download the baseline model for comparison
sbatch oellm/baseline_repro/evaluations/download_baseline.sh

# Manual: Serve model with chat UI
sbatch oellm/baseline_repro/evaluations/manual/serve_model.sh instruct

# Benchmarks: Run automated evaluation (requires slurmpilot setup)
cd oellm/baseline_repro/evaluations/benchmarks
python launch_evaluation.py
```

## Option 1: Direct Testing with OLMo-core

Test models directly using the native OLMo-core checkpoint format:

```bash
cd /work/dlclarge2/ferreira-oellm/open-instruct

# Test instruct model
srun -p alldlc2_gpu-h200 --gpus=1 --time=1:00:00 \
  /work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/python \
  oellm/baseline_repro/test_model.py \
  checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft/step3394 \
  --max-new-tokens 256 --name instruct

# Test think model (use longer generation for reasoning)
srun -p alldlc2_gpu-h200 --gpus=1 --time=1:00:00 \
  /work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/python \
  oellm/baseline_repro/test_model.py \
  checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft/step43376 \
  --max-new-tokens 512 --name think
```

Output files are saved to `oellm/baseline_repro/test_output_*.txt`.

## Option 2: HuggingFace Format (for vLLM/transformers)

### Convert Checkpoints

Run the conversion script:

```bash
cd /work/dlclarge2/ferreira-oellm/open-instruct

# Convert instruct model
sbatch oellm/baseline_repro/convert_to_hf.sh instruct

# Convert think model
sbatch oellm/baseline_repro/convert_to_hf.sh think
```

Converted models are saved to:
- `checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf/`
- `checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf/`

### Use with transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Use with vLLM

```bash
srun -p alldlc2_gpu-h200 --gpus=1 --time=1:00:00 \
  /work/dlclarge2/ferreira-oellm/open-instruct/.venv/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf \
  --trust-remote-code \
  --port 8000
```

Then query the API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dolci-instruct-sft-hf",
    "messages": [{"role": "user", "content": "What is machine learning?"}]
  }'
```

## Option 3: Interactive Chat UI (Recommended)

Serve the model with vLLM and a browser-based chat interface.

> **Note**: Chat UI tools have moved to [`evaluations/manual/`](evaluations/manual/). See the [manual evaluation README](evaluations/manual/README.md) for full details.

```bash
cd /work/dlclarge2/ferreira-oellm/open-instruct

# Serve instruct model
sbatch oellm/baseline_repro/evaluations/manual/serve_model.sh instruct

# Serve think model
sbatch oellm/baseline_repro/evaluations/manual/serve_model.sh think

# Serve baseline for comparison
sbatch oellm/baseline_repro/evaluations/manual/serve_model.sh baseline
```

Check the log for connection instructions:

```bash
tail -f oellm/baseline_repro/logs/serve_*.log
```

### Manual Setup (separate terminals)

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
python oellm/baseline_repro/evaluations/manual/chat_ui.py --api-base http://localhost:8000 --model dolci-instruct --share
```

## Model Configuration

Both models are OLMo-3 7B with:
- **Sequence length**: 32,768 tokens
- **Tokenizer**: `allenai/dolma2-tokenizer`
- **Architecture**: `Olmo3ForCausalLM` with sliding window attention
- **Precision**: bfloat16

## Files in this Directory

- `test_model.py` - Script to test models with sample prompts
- `convert_to_hf.sh` - SLURM script to convert checkpoints to HuggingFace format
- `test_output_instruct.txt` - Sample outputs from instruct model
- `test_output_think.txt` - Sample outputs from think model
- `MODEL_TESTING_GUIDE.md` - This file
- `logs/` - SLURM job output logs (serve, convert, download)
- `evaluations/` - Model evaluation tools
  - `download_baseline.sh` - Download official baseline model
  - `manual/` - Interactive chat testing (vLLM + Gradio)
  - `benchmarks/` - Automated LLM-judge evaluation

