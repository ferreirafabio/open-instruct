#!/usr/bin/env python3
"""
Launch LLM-Judge evaluation directly on the cluster using sbatch.
(Alternative to launch_eval.py which uses slurmpilot for remote submission)

Usage:
    python launch_eval_local.py
"""

import os
import subprocess
import tempfile
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/work/dlclarge2/ferreira-oellm/open-instruct")
OPENJURY_DIR = PROJECT_ROOT / "oellm/evaluations/benchmarks/OpenJury"
PYTHON_BINARY = str(PROJECT_ROOT / ".venv/bin/python")
SYMLINK_DIR = PROJECT_ROOT / "models/eval"

# Create symlinks with short names to avoid filename-too-long errors
SYMLINK_DIR.mkdir(parents=True, exist_ok=True)

baseline_path = PROJECT_ROOT / "models/baselines/Olmo-3-7B-Instruct-SFT"
model_path = PROJECT_ROOT / "checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf"

# Create/update symlinks
baseline_symlink = SYMLINK_DIR / "instruct-baseline"
model_symlink = SYMLINK_DIR / "instruct-ours"
baseline_symlink.unlink(missing_ok=True)
model_symlink.unlink(missing_ok=True)
os.symlink(baseline_path, baseline_symlink)
os.symlink(model_path, model_symlink)

# Use short symlink paths
baseline = f"VLLM/{baseline_symlink}"
datasets = [
    "alpaca-eval",
    "arena-hard",
    "m-arena-hard-EU",
]
models = [
    f"VLLM/{model_symlink}",
]

judge_model = "VLLM/Qwen/Qwen3-30B-A3B-Instruct-2507"
jobname = "olmo3-instruct-sft"

# Build all job configs
job_configs = [
    {
        "dataset": dataset,
        "model_A": baseline,
        "model_B": model,
        "judge_model": judge_model,
        "n_instructions": 50000,
    }
    for model in models
    for dataset in datasets
]

print(f"Launching {len(job_configs)} evaluation job(s)...")

for i, config in enumerate(job_configs):
    # Build the command arguments
    args = [
        f"--dataset={config['dataset']}",
        f"--model_A={config['model_A']}",
        f"--model_B={config['model_B']}",
        f"--judge_model={config['judge_model']}",
        f"--n_instructions={config['n_instructions']}",
    ]
    
    # Create sbatch script
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=llm-judge-{jobname}-{config['dataset']}
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output={PROJECT_ROOT}/oellm/evaluations/logs/eval_%j.log

cd {OPENJURY_DIR}
export PYTHONPATH="{OPENJURY_DIR}:$PYTHONPATH"
export HF_HOME="{PROJECT_ROOT}/.cache/huggingface"

{PYTHON_BINARY} {OPENJURY_DIR}/openjury/generate_and_evaluate.py \\
    {' '.join(args)}
"""
    
    # Write and submit
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(sbatch_script)
        script_path = f.name
    
    print(f"\nJob {i+1}/{len(job_configs)}: {config['dataset']}")
    print(f"  Baseline: {config['model_A'].split('/')[-1]}")
    print(f"  Model: {config['model_B'].split('/')[-1]}")
    
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Submitted: {result.stdout.strip()}")
    else:
        print(f"  Error: {result.stderr}")
    
    # Clean up temp file
    Path(script_path).unlink()

print("\nAll jobs submitted!")
print(f"Monitor with: squeue -u $USER")
print(f"After completion: python analyse_results.py --type instruct")

