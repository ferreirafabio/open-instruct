#!/usr/bin/env python3
"""Reusable SSH helper for connecting to HoreKa via aadlogin proxy with OTP + password.

Usage:
    # Establish persistent tunnel (needs OTP):
    python ssh_horeka.py tunnel --otp 123456

    # Run a command (uses existing tunnel):
    python ssh_horeka.py run -c "hostname && whoami"
    python ssh_horeka.py run -c "sinfo -p accelerated-h200"

    # Run interactively (uses existing tunnel):
    python ssh_horeka.py shell

    # Check tunnel status:
    python ssh_horeka.py status
"""

import pexpect
import sys
import os
import subprocess
import time

HOREKA_USER = "fr_ff1042"
HOREKA_HOST = "horeka.scc.kit.edu"
PROXY_HOST = "ferreira@aadlogin.informatik.uni-freiburg.de"
CONTROL_PATH = "/tmp/ssh-mux/horeka"
AADLOGIN_CONTROL_PATH = "/tmp/ssh-mux/aadlogin"
ENV_PATH = "/work/dlclarge2/ferreira-oellm/open-instruct/.env"


def load_password():
    """Load horeka password from .env file."""
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith("HOREKA_PW="):
                return line.strip().split("=", 1)[1]
    return None


def _ensure_aadlogin_tunnel():
    """Ensure persistent connection to aadlogin exists."""
    os.makedirs("/tmp/ssh-mux", exist_ok=True)
    result = subprocess.run(
        ["ssh", "-O", "check", "-o", f"ControlPath={AADLOGIN_CONTROL_PATH}", PROXY_HOST],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return True
    subprocess.run([
        "ssh", "-o", "ControlMaster=yes", "-o", f"ControlPath={AADLOGIN_CONTROL_PATH}",
        "-o", "ControlPersist=yes", "-fN", PROXY_HOST
    ], check=True)
    return True


def tunnel_active():
    """Check if horeka tunnel is active."""
    result = subprocess.run(
        ["ssh", "-O", "check", "-o", f"ControlPath={CONTROL_PATH}",
         f"{HOREKA_USER}@{HOREKA_HOST}"],
        capture_output=True, text=True
    )
    return result.returncode == 0


def establish_tunnel(otp, password=None):
    """Establish persistent ControlMaster SSH tunnel to horeka using OTP + password.

    The tunnel persists indefinitely (ControlPersist=yes) so subsequent
    ssh commands reuse it without needing OTP again.
    """
    if password is None:
        password = load_password()

    _ensure_aadlogin_tunnel()

    if tunnel_active():
        print("Horeka tunnel already active.")
        return True

    # Use pexpect to handle OTP + password prompts, then background a ControlMaster
    proxy_cmd = f"ssh -o ControlPath={AADLOGIN_CONTROL_PATH} -W %h:%p {PROXY_HOST}"
    cmd = (
        f'ssh -o ControlMaster=yes -o "ControlPath={CONTROL_PATH}" '
        f'-o ControlPersist=yes '
        f'-o "ProxyCommand={proxy_cmd}" '
        f'-o ConnectTimeout=15 '
        f'-N {HOREKA_USER}@{HOREKA_HOST}'
    )

    child = pexpect.spawn("/bin/bash", ["-c", cmd], timeout=30, encoding="utf-8")

    try:
        child.expect("OTP", timeout=20)
        child.sendline(otp)

        idx = child.expect(["[Pp]assword", "OTP"], timeout=15)
        if idx == 1:
            print("OTP rejected (expired or invalid).", file=sys.stderr)
            child.close()
            return False

        child.sendline(password)

        # -N means no command, so SSH just stays open. Wait a moment then check.
        time.sleep(3)

        if tunnel_active():
            print("Horeka tunnel established successfully. Use 'run' or 'shell' commands.")
            # Detach: the ControlMaster runs in background due to ControlPersist
            return True
        else:
            # Wait a bit more and retry
            time.sleep(3)
            if tunnel_active():
                print("Horeka tunnel established successfully.")
                return True
            print("Tunnel failed to establish.", file=sys.stderr)
            child.close()
            return False
    except pexpect.TIMEOUT:
        print(f"Timeout during tunnel setup.", file=sys.stderr)
        child.close()
        return False
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        child.close()
        return False


def run_on_horeka(command, timeout=120):
    """Run a command on horeka via existing tunnel. Returns (stdout, stderr)."""
    if not tunnel_active():
        return None, "No active tunnel. Run: python ssh_horeka.py tunnel --otp <OTP>"

    result = subprocess.run(
        [
            "ssh", "-o", f"ControlPath={CONTROL_PATH}",
            f"{HOREKA_USER}@{HOREKA_HOST}",
            command
        ],
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout, result.stderr


def interactive_shell():
    """Open an interactive shell on horeka via existing tunnel."""
    if not tunnel_active():
        print("No active tunnel. Run: python ssh_horeka.py tunnel --otp <OTP>")
        return
    os.execvp("ssh", [
        "ssh", "-o", f"ControlPath={CONTROL_PATH}",
        f"{HOREKA_USER}@{HOREKA_HOST}"
    ])


def close_tunnel():
    """Close the horeka tunnel."""
    subprocess.run([
        "ssh", "-O", "exit", "-o", f"ControlPath={CONTROL_PATH}",
        f"{HOREKA_USER}@{HOREKA_HOST}"
    ], capture_output=True)
    print("Tunnel closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SSH helper for HoreKa")
    sub = parser.add_subparsers(dest="action", help="Action to perform")

    p_tunnel = sub.add_parser("tunnel", help="Establish persistent tunnel")
    p_tunnel.add_argument("--otp", required=True, help="TOTP code")

    p_run = sub.add_parser("run", help="Run command on horeka")
    p_run.add_argument("-c", "--command", required=True, help="Command to run")
    p_run.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")

    p_shell = sub.add_parser("shell", help="Open interactive shell")
    p_status = sub.add_parser("status", help="Check tunnel status")
    p_close = sub.add_parser("close", help="Close tunnel")

    args = parser.parse_args()

    if args.action == "tunnel":
        ok = establish_tunnel(args.otp)
        sys.exit(0 if ok else 1)
    elif args.action == "run":
        stdout, stderr = run_on_horeka(args.command, timeout=args.timeout)
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        sys.exit(0 if stdout is not None else 1)
    elif args.action == "shell":
        interactive_shell()
    elif args.action == "status":
        if tunnel_active():
            print("Tunnel is ACTIVE")
        else:
            print("Tunnel is DOWN")
    elif args.action == "close":
        close_tunnel()
    else:
        parser.print_help()
