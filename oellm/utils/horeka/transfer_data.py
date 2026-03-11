#!/usr/bin/env python3
"""Transfer training data from kislurm to horeka workspace with auto-OTP."""

import pexpect, sys, time, re, pyotp, os

ENV_PATH = "/work/dlclarge2/ferreira-oellm/open-instruct/.env"
PROXY = "ferreira@aadlogin.informatik.uni-freiburg.de"
AADLOGIN_CTL = "/tmp/ssh-mux/aadlogin"
HOREKA_USER = "fr_ff1042"
HOREKA_HOST = "horeka.scc.kit.edu"
WS_REMOTE = "/hkfs/work/workspace/scratch/fr_ff1042-oellm/open-instruct"


def load_env():
    env = {}
    with open(ENV_PATH) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=', 1)
                env[k] = v
    return env


_last_otp = None

def fresh_otp(env):
    global _last_otp
    totp = pyotp.TOTP(env['HOREKA_TOTP_SECRET'])
    # Always ensure we get a different OTP than last time
    otp = totp.now()
    if otp == _last_otp:
        remaining = totp.interval - (int(time.time()) % totp.interval)
        print(f"  Waiting {remaining+1}s for fresh OTP window (same token as last call)...")
        time.sleep(remaining + 1)
        otp = totp.now()
    # Also wait if not enough time left for the connection to use the OTP
    remaining = totp.interval - (int(time.time()) % totp.interval)
    if remaining < 10:
        print(f"  Waiting {remaining+1}s for fresh OTP window...")
        time.sleep(remaining + 1)
        otp = totp.now()
    _last_otp = otp
    return otp


def rsync_to_horeka(local_path, remote_path, env, extra_args=""):
    """Run rsync with pexpect to handle OTP + password auth."""
    otp = fresh_otp(env)

    ssh_cmd = f'ssh -o ProxyCommand="ssh -o ControlPath={AADLOGIN_CTL} -W %h:%p {PROXY}"'
    cmd = f'rsync -avz --progress {extra_args} -e \'{ssh_cmd}\' {local_path} {HOREKA_USER}@{HOREKA_HOST}:{remote_path}'

    print(f"\n  rsync {local_path} -> {remote_path}")
    child = pexpect.spawn('/bin/bash', ['-c', cmd], timeout=7200, encoding='utf-8')

    # Handle OTP
    child.expect('OTP', timeout=20)
    child.sendline(otp)
    idx = child.expect(['[Pp]assword', 'OTP'], timeout=15)
    if idx == 1:
        print("  OTP REJECTED!")
        child.close()
        return False
    child.sendline(env['HOREKA_PW'])

    # Stream output
    while True:
        try:
            line = child.readline()
            if not line:
                break
            line = line.strip()
            if line:
                # Only print progress lines and summaries
                if '%' in line or 'sent' in line or 'total' in line or 'speedup' in line:
                    print(f"  {line}")
        except pexpect.EOF:
            break
        except pexpect.TIMEOUT:
            break

    child.close()
    return child.exitstatus == 0


def run_remote(cmd, env, timeout=60):
    """Run a command on horeka."""
    otp = fresh_otp(env)
    ssh = (
        f'ssh -o ConnectTimeout=15 '
        f'-o ProxyCommand="ssh -o ControlPath={AADLOGIN_CTL} -W %h:%p {PROXY}" '
        f'{HOREKA_USER}@{HOREKA_HOST} \'{cmd}\''
    )
    child = pexpect.spawn('/bin/bash', ['-c', ssh], timeout=timeout, encoding='utf-8')
    child.expect('OTP', timeout=15)
    child.sendline(otp)
    child.expect('[Pp]assword', timeout=10)
    child.sendline(env['HOREKA_PW'])
    child.expect(pexpect.EOF, timeout=timeout)
    output = child.before
    clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]|\[\?[0-9]+[hl]', '', output)
    child.close()
    return clean.strip()


if __name__ == '__main__':
    env = load_env()

    print("=== Creating directories on horeka ===")
    out = run_remote(
        f'mkdir -p {WS_REMOTE}/data/baseline_reproduction '
        f'{WS_REMOTE}/models/baselines '
        f'{WS_REMOTE}/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2 '
        f'{WS_REMOTE}/../OLMo-core && echo DONE',
        env
    )
    print(out)

    transfers = [
        {
            'name': 'OLMo-core code (47MB)',
            'local': '/work/dlclarge2/ferreira-oellm/OLMo-core/',
            'remote': f'{WS_REMOTE}/../OLMo-core/',
        },
        {
            'name': 'dolci_instruct_sft_tokenized_v2 (8GB)',
            'local': '/work/dlclarge2/ferreira-oellm/open-instruct/data/baseline_reproduction/dolci_instruct_sft_tokenized_v2/',
            'remote': f'{WS_REMOTE}/data/baseline_reproduction/dolci_instruct_sft_tokenized_v2/',
        },
        {
            'name': 'Base model checkpoint (28GB)',
            'local': '/work/dlclarge2/ferreira-oellm/open-instruct/models/baselines/Olmo-3-1025-7B-olmocore/',
            'remote': f'{WS_REMOTE}/models/baselines/Olmo-3-1025-7B-olmocore/',
        },
        {
            'name': 'dolci_think_sft_tokenized_v2 (105GB)',
            'local': '/work/dlclarge2/ferreira-oellm/open-instruct/data/baseline_reproduction/dolci_think_sft_tokenized_v2/',
            'remote': f'{WS_REMOTE}/data/baseline_reproduction/dolci_think_sft_tokenized_v2/',
        },
        {
            'name': 'Latest checkpoint step35250 (82GB)',
            'local': '/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2/step35250/',
            'remote': f'{WS_REMOTE}/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2/step35250/',
        },
    ]

    for i, t in enumerate(transfers):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(transfers)}] {t['name']}")
        print(f"{'='*60}")
        ok = rsync_to_horeka(t['local'], t['remote'], env)
        if ok:
            print(f"  OK!")
        else:
            print(f"  FAILED! Continuing with next transfer...")

    print("\n=== Verifying transfers ===")
    out = run_remote(
        f'echo "=DATA="; ls {WS_REMOTE}/data/baseline_reproduction/; '
        f'echo "=MODEL="; ls {WS_REMOTE}/models/baselines/; '
        f'echo "=CKPT="; ls {WS_REMOTE}/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2/; '
        f'echo "=OLMO="; ls {WS_REMOTE}/../OLMo-core/src/ | head -5',
        env
    )
    print(out)
    print("\n=== ALL TRANSFERS COMPLETE ===")
