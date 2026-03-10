#!/usr/bin/env python3
"""Transfer Track B tokenized data + base checkpoint to HoreKa.

Transfers:
1. Tokenized B1-90en and B2-80en datasets
2. Base checkpoint (dolci-instruct-sft-v2-horeka) — if not already on HoreKa
3. Training script + configs
4. OLMo-core code (latest)

Usage:
    python oellm/horeka/transfer_multilingual_to_horeka.py
    python oellm/horeka/transfer_multilingual_to_horeka.py --skip-checkpoint  # if already on HoreKa
    python oellm/horeka/transfer_multilingual_to_horeka.py --dry-run
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oellm.horeka.transfer_data import load_env, rsync_to_horeka, run_remote

WS_REMOTE = "/hkfs/work/workspace/scratch/fr_ff1042-oellm/open-instruct"
LOCAL = "/work/dlclarge2/ferreira-oellm/open-instruct"


def main():
    parser = argparse.ArgumentParser(description="Transfer multilingual data to HoreKa")
    parser.add_argument("--skip-checkpoint", action="store_true", default=True,
                        help="Skip base checkpoint transfer (already on HoreKa at fr_ff1042 path)")
    parser.add_argument("--transfer-checkpoint", action="store_true",
                        help="Force transfer base checkpoint even if it exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be transferred")
    parser.add_argument("--experiments", nargs="+", default=["B1-90en", "B2-80en"],
                        help="Which experiments to transfer (default: B1-90en B2-80en)")
    args = parser.parse_args()

    env = load_env()

    transfers = []

    # 1. Tokenized datasets
    for exp in args.experiments:
        local_path = f"{LOCAL}/data/datasets_multilingual_sft/tokenized/{exp}/"
        remote_path = f"{WS_REMOTE}/data/datasets_multilingual_sft/tokenized/{exp}/"
        transfers.append({
            "name": f"Tokenized {exp}",
            "local": local_path,
            "remote": remote_path,
        })

    # 2. Base checkpoint (the instruct checkpoint we continue from)
    # Already on HoreKa at checkpoints/fr_ff1042/olmo3-7b-sft/dolci-instruct-sft-v2-horeka/
    if args.transfer_checkpoint:
        transfers.append({
            "name": "Base checkpoint (dolci-instruct-sft-v2-horeka)",
            "local": f"{LOCAL}/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka/",
            "remote": f"{WS_REMOTE}/checkpoints/fr_ff1042/olmo3-7b-sft/dolci-instruct-sft-v2-horeka/",
        })

    # 3. Training script + configs
    transfers.append({
        "name": "HoreKa training script",
        "local": f"{LOCAL}/oellm/horeka/train_multilingual_sft_horeka.sh",
        "remote": f"{WS_REMOTE}/oellm/horeka/train_multilingual_sft_horeka.sh",
    })
    transfers.append({
        "name": "Multilingual configs",
        "local": f"{LOCAL}/oellm/configs/",
        "remote": f"{WS_REMOTE}/oellm/configs/",
    })

    # 4. DeepSpeed config (needed by training script)
    transfers.append({
        "name": "DeepSpeed config",
        "local": f"{LOCAL}/configs/ds_configs/",
        "remote": f"{WS_REMOTE}/configs/ds_configs/",
    })

    # 5. OLMo-core code (latest)
    transfers.append({
        "name": "OLMo-core code",
        "local": "/work/dlclarge2/ferreira-oellm/OLMo-core/",
        "remote": f"{WS_REMOTE}/../OLMo-core/",
    })

    if args.dry_run:
        print("=== DRY RUN — would transfer: ===")
        for t in transfers:
            print(f"  {t['name']}")
            print(f"    {t['local']} -> {t['remote']}")
        return

    # Create directories on HoreKa
    print("=== Creating directories on HoreKa ===")
    dirs = set()
    for t in transfers:
        remote = t["remote"]
        if remote.endswith("/"):
            dirs.add(remote)
        else:
            dirs.add(os.path.dirname(remote))
    mkdir_cmd = "mkdir -p " + " ".join(dirs) + " && echo DONE"
    out = run_remote(mkdir_cmd, env)
    print(out)

    # Transfer
    for i, t in enumerate(transfers):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(transfers)}] {t['name']}")
        print(f"{'='*60}")
        ok = rsync_to_horeka(t["local"], t["remote"], env)
        if ok:
            print("  OK!")
        else:
            print("  FAILED! Continuing with next transfer...")

    # Verify
    print("\n=== Verifying transfers ===")
    verify_cmds = []
    for exp in args.experiments:
        verify_cmds.append(f'echo "={exp}="; ls {WS_REMOTE}/data/datasets_multilingual_sft/tokenized/{exp}/ 2>/dev/null || echo "MISSING"')
    if not args.skip_checkpoint:
        verify_cmds.append(f'echo "=CKPT="; ls {WS_REMOTE}/checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-v2-horeka/ 2>/dev/null | head -5 || echo "MISSING"')
    verify_cmds.append(f'echo "=SCRIPT="; ls {WS_REMOTE}/oellm/horeka/train_multilingual_sft_horeka.sh 2>/dev/null || echo "MISSING"')

    out = run_remote("; ".join(verify_cmds), env, timeout=30)
    print(out)

    print("\n=== ALL TRANSFERS COMPLETE ===")
    print("\nTo submit training on HoreKa:")
    for exp in args.experiments:
        print(f"  EXPERIMENT={exp} sbatch oellm/horeka/train_multilingual_sft_horeka.sh")


if __name__ == "__main__":
    main()
