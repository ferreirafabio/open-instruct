#!/usr/bin/env python3
"""Transfer checkpoints from HoreKa to kislurm with auto-OTP and checksum verification.

Transfers the latest checkpoint first, then the rest in reverse order.
Verifies checksums (md5) after each checkpoint transfer.
"""

import pexpect, time, re, pyotp, os, sys, subprocess, hashlib

ENV_PATH = "/work/dlclarge2/ferreira-oellm/open-instruct/.env"
PROXY = "ferreira@aadlogin.informatik.uni-freiburg.de"
AADLOGIN_CTL = "/tmp/ssh-mux/aadlogin"
HOREKA_USER = "fr_ff1042"
HOREKA_HOST = "horeka.scc.kit.edu"
WS_REMOTE = "/hkfs/work/workspace/scratch/fr_ff1042-oellm/open-instruct"

LOCAL_DEST = "/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2-horeka"
REMOTE_CKPT = f"{WS_REMOTE}/checkpoints/fr_ff1042/olmo3-7b-sft/dolci-think-sft-v2-horeka"

# Skip: symlink, wandb, and checkpoints already on kislurm (dolci-think-sft-v2 has up to step36450)
KISLURM_CKPT = "/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-v2"
SKIP = {"step35250", "wandb"}
# Also skip any step dirs that already exist in the kislurm checkpoint dir
if os.path.isdir(KISLURM_CKPT):
    SKIP.update(d for d in os.listdir(KISLURM_CKPT) if d.startswith("step"))


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
    # Always wait until we get a code different from the last one used
    while True:
        remaining = totp.interval - (int(time.time()) % totp.interval)
        code = totp.now()
        if code != _last_otp and remaining >= 10:
            _last_otp = code
            return code
        wait = min(remaining + 1, 31)
        print(f"  Waiting {wait}s for fresh OTP (avoiding reuse)...", flush=True)
        time.sleep(wait)


def rsync_from_horeka(remote_path, local_path, env, timeout=14400):
    """rsync FROM horeka TO local with --checksum for integrity."""
    otp = fresh_otp(env)

    ssh_cmd = f'ssh -o ProxyCommand="ssh -o ControlPath={AADLOGIN_CTL} -W %h:%p {PROXY}"'
    cmd = f'rsync -avz --checksum --progress -e \'{ssh_cmd}\' {HOREKA_USER}@{HOREKA_HOST}:{remote_path} {local_path}'

    print(f"\n  rsync {remote_path} -> {local_path}", flush=True)
    child = pexpect.spawn('/bin/bash', ['-c', cmd], timeout=timeout, encoding='utf-8')

    child.expect('OTP', timeout=30)
    child.sendline(otp)
    idx = child.expect(['[Pp]assword', 'OTP'], timeout=15)
    if idx == 1:
        print("  OTP REJECTED!", flush=True)
        child.close()
        return False
    child.sendline(env['HOREKA_PW'])

    last_print = time.time()
    while True:
        try:
            line = child.readline()
            if not line:
                break
            line = line.strip()
            if line and time.time() - last_print > 10:
                if '%' in line or 'sent' in line or 'total' in line or 'speedup' in line:
                    print(f"  {line}", flush=True)
                    last_print = time.time()
        except pexpect.EOF:
            break
        except pexpect.TIMEOUT:
            break

    child.close()
    return child.exitstatus == 0


def run_remote(cmd, env, timeout=120):
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


def get_remote_checksums(env, step_dir):
    """Get md5 checksums for all files in a remote checkpoint dir."""
    cmd = f'find {REMOTE_CKPT}/{step_dir} -type f | sort | xargs md5sum'
    out = run_remote(cmd, env, timeout=300)
    checksums = {}
    for line in out.split('\n'):
        line = line.strip()
        if '  ' in line:
            md5, path = line.split('  ', 1)
            # Use relative path from checkpoint dir
            rel = path.split(f'{step_dir}/')[-1] if f'{step_dir}/' in path else path
            checksums[rel] = md5
    return checksums


def get_local_checksums(local_dir):
    """Get md5 checksums for all files in a local checkpoint dir."""
    checksums = {}
    for root, _, files in os.walk(local_dir):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, local_dir)
            md5 = hashlib.md5()
            with open(fpath, 'rb') as fh:
                for chunk in iter(lambda: fh.read(8192 * 1024), b''):
                    md5.update(chunk)
            checksums[rel] = md5.hexdigest()
    return checksums


def verify_checksums(env, step_dir):
    """Compare remote and local checksums for a checkpoint."""
    print(f"  Verifying checksums for {step_dir}...", flush=True)
    remote = get_remote_checksums(env, step_dir)
    local_dir = os.path.join(LOCAL_DEST, step_dir)
    local = get_local_checksums(local_dir)

    if not remote:
        print(f"  WARNING: No remote checksums retrieved!", flush=True)
        return False

    ok = True
    for rel, rmd5 in remote.items():
        lmd5 = local.get(rel)
        if lmd5 is None:
            print(f"  MISSING locally: {rel}", flush=True)
            ok = False
        elif lmd5 != rmd5:
            print(f"  MISMATCH: {rel} (remote={rmd5}, local={lmd5})", flush=True)
            ok = False

    for rel in local:
        if rel not in remote:
            print(f"  EXTRA locally: {rel}", flush=True)

    if ok:
        print(f"  CHECKSUMS OK ({len(remote)} files verified)", flush=True)
    return ok


def prioritize_dirs(dirs):
    """Sort dirs so latest checkpoint comes first, then descending."""
    step_dirs = []
    other_dirs = []
    for d in dirs:
        if d.startswith('step'):
            try:
                num = int(d.replace('step', ''))
                step_dirs.append((num, d))
            except ValueError:
                other_dirs.append(d)
        else:
            other_dirs.append(d)
    # Sort steps descending (latest first)
    step_dirs.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in step_dirs] + other_dirs


if __name__ == '__main__':
    env = load_env()

    print("=== Listing checkpoints on HoreKa ===", flush=True)
    out = run_remote(f'ls -1 {REMOTE_CKPT}/', env)
    print(out, flush=True)

    raw_dirs = [d.strip() for d in out.split('\n') if d.strip() and d.strip() not in SKIP]
    # Filter out non-directory entries (like ':' from ls output)
    dirs = [d for d in raw_dirs if d.startswith('step') or d == 'tokenizer']
    dirs = prioritize_dirs(dirs)
    print(f"\nWill transfer {len(dirs)} items (latest first): {dirs}", flush=True)
    print(f"Skipping: {SKIP}", flush=True)

    os.makedirs(LOCAL_DEST, exist_ok=True)

    failed = []
    for i, d in enumerate(dirs):
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(dirs)}] {d}", flush=True)
        print(f"{'='*60}", flush=True)

        local_d = os.path.join(LOCAL_DEST, d)
        if os.path.exists(local_d):
            print(f"  Already exists locally. Verifying checksums...", flush=True)
            if d.startswith('step'):
                ok = verify_checksums(env, d)
                if ok:
                    print(f"  Skipping (already verified).", flush=True)
                    continue
                else:
                    print(f"  Checksum mismatch! Re-transferring...", flush=True)
                    subprocess.run(['rm', '-rf', local_d])
            else:
                print(f"  Skipping non-step dir.", flush=True)
                continue

        ok = rsync_from_horeka(
            f'{REMOTE_CKPT}/{d}/',
            f'{LOCAL_DEST}/{d}/',
            env
        )
        if ok:
            print(f"  Transfer OK!", flush=True)
            # Verify checksums after transfer
            if d.startswith('step'):
                checksum_ok = verify_checksums(env, d)
                if not checksum_ok:
                    print(f"  CHECKSUM VERIFICATION FAILED for {d}!", flush=True)
                    failed.append(d)
        else:
            print(f"  TRANSFER FAILED for {d}!", flush=True)
            failed.append(d)

    print(f"\n{'='*60}", flush=True)
    print("=== Summary ===", flush=True)
    print(f"{'='*60}", flush=True)
    for d in sorted(os.listdir(LOCAL_DEST)):
        path = os.path.join(LOCAL_DEST, d)
        if os.path.isdir(path):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(path)
                for f in fns
            ) / (1024**3)
            print(f"  {d}: {size:.1f} GB", flush=True)

    if failed:
        print(f"\nFAILED: {failed}", flush=True)
        print("Re-run this script to retry failed transfers.", flush=True)
    else:
        print("\nALL TRANSFERS AND CHECKSUMS VERIFIED!", flush=True)
