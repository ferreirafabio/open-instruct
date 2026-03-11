#!/usr/bin/env python3
"""Transfer ELO evaluation dependencies to HoreKa.

Transfers:
1. OpenJury code (oellm/evaluations/benchmarks/OpenJury/)
2. OpenJury eval data (data/openjury-eval-data/)
3. Model checkpoint (A1-90en HF)
4. Qwen judge model (from HF cache)
5. ELO eval script
6. Model eval symlinks

Usage:
    python oellm/utils/horeka/transfer_elo_eval_to_horeka.py
    python oellm/utils/horeka/transfer_elo_eval_to_horeka.py --dry-run
    python oellm/utils/horeka/transfer_elo_eval_to_horeka.py --skip-judge  # if judge already on HoreKa
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from oellm.utils.horeka.transfer_data import load_env, rsync_to_horeka, run_remote

WS_REMOTE = "/hkfs/work/workspace/scratch/fr_ff1042-oellm/open-instruct"
LOCAL = "/work/dlclarge2/ferreira-oellm/open-instruct"


def main():
    parser = argparse.ArgumentParser(description="Transfer ELO eval to HoreKa")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip Qwen judge model transfer")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip eval model checkpoint transfer")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be transferred")
    args = parser.parse_args()

    env = load_env()

    transfers = []

    # 1. OpenJury code
    transfers.append({
        "name": "OpenJury code",
        "local": f"{LOCAL}/oellm/evaluations/benchmarks/OpenJury/",
        "remote": f"{WS_REMOTE}/oellm/evaluations/benchmarks/OpenJury/",
        "extra": "--exclude '.git' --exclude '__pycache__' --exclude 'results'",
    })

    # 2. OpenJury eval data (LMArena battles, m-ArenaHard)
    transfers.append({
        "name": "OpenJury eval data",
        "local": f"{LOCAL}/data/openjury-eval-data/",
        "remote": f"{WS_REMOTE}/data/openjury-eval-data/",
        "extra": "--exclude 'cache'",
    })

    # 3. Eval model checkpoint (A1-90en)
    if not args.skip_model:
        transfers.append({
            "name": "A1-90en HF checkpoint",
            "local": f"{LOCAL}/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-eu-A1-90en-hf/",
            "remote": f"{WS_REMOTE}/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-eu-A1-90en-hf/",
            "extra": "",
        })

    # 4. Qwen judge model
    if not args.skip_judge:
        transfers.append({
            "name": "Qwen3-30B-A3B judge model",
            "local": f"{LOCAL}/models/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/",
            "remote": f"{WS_REMOTE}/models/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/",
            "extra": "",
        })

    # 5. ELO eval script
    transfers.append({
        "name": "ELO eval script",
        "local": f"{LOCAL}/oellm/evaluations/benchmarks/run_elo_evaluation_horeka_h200.sh",
        "remote": f"{WS_REMOTE}/oellm/evaluations/benchmarks/",
        "extra": "",
    })

    print(f"\n{'='*60}")
    print("ELO EVAL TRANSFER TO HOREKA")
    print(f"{'='*60}")
    for t in transfers:
        print(f"  [{t['name']}] {t['local']} -> {t['remote']}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("DRY RUN: no transfers performed.")
        return

    # Create remote directories
    dirs = [
        f"{WS_REMOTE}/oellm/evaluations/benchmarks",
        f"{WS_REMOTE}/oellm/evaluations/logs",
        f"{WS_REMOTE}/data/openjury-eval-data",
        f"{WS_REMOTE}/models/huggingface/hub",
        f"{WS_REMOTE}/models/eval",
        f"{WS_REMOTE}/checkpoints/ferreira/olmo3-7b-sft",
    ]
    print("Creating remote directories...")
    run_remote(f"mkdir -p {' '.join(dirs)}", env)

    # Create eval symlink on remote
    print("Creating eval symlink on remote...")
    run_remote(
        f"ln -sfn {WS_REMOTE}/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-eu-A1-90en-hf/step238 "
        f"{WS_REMOTE}/models/eval/eu-A1-90en-step238",
        env
    )

    # Transfer each item
    for t in transfers:
        print(f"\n--- Transferring: {t['name']} ---")
        ok = rsync_to_horeka(t["local"], t["remote"], env, extra_args=t.get("extra", ""))
        if not ok:
            print(f"FAILED: {t['name']}")
            sys.exit(1)
        print(f"OK: {t['name']}")

    # Install OpenJury + vllm on HoreKa
    print("\n--- Installing OpenJury + vllm on HoreKa ---")
    install_cmd = (
        f"source {WS_REMOTE}/.venv/bin/activate && "
        f"cd {WS_REMOTE}/oellm/evaluations/benchmarks/OpenJury && "
        f"pip install -e '.[vllm]' 2>&1 | tail -5"
    )
    out = run_remote(install_cmd, env, timeout=600)
    print(out)

    print(f"\n{'='*60}")
    print("All transfers complete!")
    print(f"\nTo submit on HoreKa:")
    print(f"  cd {WS_REMOTE}")
    print(f"  sbatch oellm/evaluations/benchmarks/run_elo_evaluation_horeka_h200.sh eu-A1-90en-step238")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
