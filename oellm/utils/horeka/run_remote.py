#!/usr/bin/env python3
"""Run a script/commands on HoreKa via SSH with auto-generated OTP.

Usage:
    python run_remote.py "hostname && whoami"
    python run_remote.py "sinfo" --timeout 30
"""

import pexpect, sys, time, re, pyotp

ENV_PATH = "/work/dlclarge2/ferreira-oellm/open-instruct/.env"
PROXY = "ferreira@aadlogin.informatik.uni-freiburg.de"
AADLOGIN_CTL = "/tmp/ssh-mux/aadlogin"


def load_env():
    env = {}
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env[k] = v
    return env


def connect():
    env = load_env()
    totp = pyotp.TOTP(env['HOREKA_TOTP_SECRET'])
    remaining = totp.interval - (int(time.time()) % totp.interval)
    if remaining < 15:
        time.sleep(remaining + 1)
    otp = totp.now()

    cmd = (
        f'ssh -o ConnectTimeout=15 -tt '
        f'-o ProxyCommand="ssh -o ControlPath={AADLOGIN_CTL} -W %h:%p {PROXY}" '
        f'fr_ff1042@horeka.scc.kit.edu "bash --norc --noprofile"'
    )
    child = pexpect.spawn(cmd, timeout=60, encoding='utf-8')
    child.expect('OTP', timeout=15)
    child.sendline(otp)
    i = child.expect(['[Pp]assword', 'OTP'], timeout=10)
    if i == 1:
        raise RuntimeError("OTP rejected")
    child.sendline(env['HOREKA_PW'])
    time.sleep(3)
    try:
        child.read_nonblocking(size=20000, timeout=3)
    except:
        pass
    # Set clean prompt
    child.sendline('export PS1="H> "')
    time.sleep(1)
    try:
        child.read_nonblocking(size=5000, timeout=1)
    except:
        pass
    return child


def run_cmd(child, cmd, timeout=30):
    MARKER = 'XYZENDXYZ'
    child.sendline(f'{cmd}; echo {MARKER}')
    child.expect(MARKER, timeout=timeout)
    raw = child.before
    # Strip ANSI escape codes
    clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]|\[\?[0-9]+[hl]', '', raw)
    lines = clean.strip().split('\r\n')
    # Skip echoed command
    return '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='Command(s) to run on horeka')
    parser.add_argument('--timeout', type=int, default=30)
    args = parser.parse_args()

    child = connect()
    result = run_cmd(child, args.command, timeout=args.timeout)
    print(result)
    child.sendline('exit')
    try:
        child.expect(pexpect.EOF, timeout=5)
    except:
        pass
