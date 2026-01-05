#!/usr/bin/env python3
"""
Launch LLM-Judge evaluation using slurmpilot.

Usage:
    # Install slurmpilot first: uv add slurmpilot
    python launch_eval.py
"""

from pathlib import Path

from slurmpilot import SlurmPilot, JobCreationInfo, unify

cluster = "kislurm"
slurm = SlurmPilot(clusters=[cluster])

# Paths
PROJECT_ROOT = Path("/work/dlclarge2/ferreira-oellm/open-instruct")

baseline = f"VLLM/{PROJECT_ROOT}/models/baselines/Olmo-3-7B-Instruct-SFT"
datasets = [
    "alpaca-eval",
    "arena-hard",
    "m-arena-hard-EU",
]
models = [
    f"VLLM/{PROJECT_ROOT}/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf",
    # Add more models here if needed:
    # f"VLLM/{PROJECT_ROOT}/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf",
]

# judge_model = "VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
# judge_model = "VLLM/RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8"
judge_model = "VLLM/Qwen/Qwen3-30B-A3B-Instruct-2507"
# judge_model = "VLLM/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
jobname = "olmo3-instruct-sft"

job_info = JobCreationInfo(
    cluster=cluster,
    partition="alldlc2_gpu-h200",
    jobname=unify(f"llm-judge-{jobname}", method="date"),
    entrypoint="generate_and_evaluate.py",
    python_binary=str(PROJECT_ROOT / ".venv/bin/python"),
    python_args=[
        {
            # "dataset": f"m-arena-hard-EU",
            "dataset": dataset,
            "model_A": baseline,
            "model_B": model,
            "judge_model": judge_model,
            "n_instructions": 50000,
            # "ignore_cache": None,
        }
        for model in models
        for dataset in datasets
    ],
    src_dir=str(Path(__file__).parent / "OpenJury/openjury/"),
    python_libraries=[str(Path(__file__).parent / "OpenJury/")],
    n_cpus=1,
    max_runtime_minutes=60 * 2,
    env={
        "HF_HOME": str(PROJECT_ROOT / ".cache/huggingface"),
        # "HF_HUB_OFFLINE": "1",
    },
)

# Launch the job
job_id = slurm.schedule_job(job_info)
print(f"Job {job_id} scheduled on {job_info.cluster}")
# slurm.wait_completion(job_info.jobname)
