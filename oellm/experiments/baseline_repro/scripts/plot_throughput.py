#!/usr/bin/env python3
"""Plot training throughput over time for OLMo-3-7B SFT baseline reproduction.

Reads wandb output.log files (or grep-extracted metric lines) and produces:
1. TPS/device over training steps (all 4 runs)
2. MFU (corrected H200) over training steps (all 4 runs)
3. Cumulative wall-clock time over steps (all 4 runs)

Usage:
    python oellm/experiments/baseline_repro/scripts/plot_throughput.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # baseline_repro/
CKPT = Path("/work/dlclarge2/ferreira-oellm/open-instruct/checkpoints/ferreira/olmo3-7b-sft")
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
EVAL_FIGURES = ROOT.parent.parent / "evaluations" / "figures"
EVAL_FIGURES.mkdir(exist_ok=True)

# MFU correction: OLMo-core used A100 peak (156 TFLOPS) instead of H200 (989.5 TFLOPS)
A100_PEAK = 156e12
H200_PEAK = 989.5e12
MFU_CORRECTION = A100_PEAK / H200_PEAK


def _parse_single_log(text: str) -> dict:
    """Parse a single output.log text for per-step metrics (logged every 10 steps)."""
    steps, tps, tps_avg, mfu, mfu_avg = [], [], [], [], []

    block_pat = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+\t.*\[step=(\d+)/\d+,epoch=\d+,eta=.*?\]'
    )
    tps_pat = re.compile(r'throughput/device/TPS=([\d,]+)')
    tps_avg_pat = re.compile(r'throughput/device/TPS \(actual avg\)=([\d,]+)')
    mfu_pat = re.compile(r'throughput/device/MFU=([\d.]+)')
    mfu_avg_pat = re.compile(r'throughput/device/MFU \(actual avg\)=([\d.]+)')

    current_step = None
    current = {}

    for line in text.split('\n'):
        bm = block_pat.search(line)
        if bm:
            if current_step is not None and 'tps' in current:
                steps.append(current_step)
                tps.append(current['tps'])
                tps_avg.append(current.get('tps_avg', current['tps']))
                mfu.append(current['mfu'])
                mfu_avg.append(current.get('mfu_avg', current['mfu']))
            current_step = int(bm.group(2))
            current = {}
            continue

        if current_step is not None:
            m = tps_pat.search(line)
            if m:
                current['tps'] = float(m.group(1).replace(',', ''))
            m = tps_avg_pat.search(line)
            if m:
                current['tps_avg'] = float(m.group(1).replace(',', ''))
            m = mfu_pat.search(line)
            if m:
                current['mfu'] = float(m.group(1))
            m = mfu_avg_pat.search(line)
            if m:
                current['mfu_avg'] = float(m.group(1))

    if current_step is not None and 'tps' in current:
        steps.append(current_step)
        tps.append(current['tps'])
        tps_avg.append(current.get('tps_avg', current['tps']))
        mfu.append(current['mfu'])
        mfu_avg.append(current.get('mfu_avg', current['mfu']))

    return {
        'steps': np.array(steps),
        'tps': np.array(tps),
        'tps_avg': np.array(tps_avg),
        'mfu': np.array(mfu),
        'mfu_avg': np.array(mfu_avg),
    }


def parse_wandb_log(log_path: Path) -> dict:
    """Parse a single output.log file."""
    data = _parse_single_log(log_path.read_text())
    data['mfu'] *= MFU_CORRECTION
    data['mfu_avg'] *= MFU_CORRECTION
    return data


def parse_wandb_multi(wandb_dir: Path) -> dict:
    """Parse ALL wandb run directories under a wandb/wandb/ dir, merging them.

    This handles training runs that were preempted and restarted multiple times
    (SLURM array=0-9%1), each creating a new wandb run directory.
    """
    all_steps, all_tps, all_tps_avg, all_mfu, all_mfu_avg = [], [], [], [], []

    run_dirs = sorted(wandb_dir.glob("*/files/output.log"))
    # Also check offline-run-* dirs
    run_dirs += sorted(wandb_dir.glob("offline-*/files/output.log"))
    run_dirs = sorted(set(run_dirs))  # dedup

    for log_path in run_dirs:
        if log_path.stat().st_size == 0:
            continue
        data = _parse_single_log(log_path.read_text())
        if len(data['steps']) > 0:
            all_steps.append(data['steps'])
            all_tps.append(data['tps'])
            all_tps_avg.append(data['tps_avg'])
            all_mfu.append(data['mfu'])
            all_mfu_avg.append(data['mfu_avg'])

    if not all_steps:
        return {'steps': np.array([]), 'tps': np.array([]),
                'tps_avg': np.array([]), 'mfu': np.array([]),
                'mfu_avg': np.array([])}

    steps = np.concatenate(all_steps)
    tps = np.concatenate(all_tps)
    tps_avg = np.concatenate(all_tps_avg)
    mfu = np.concatenate(all_mfu)
    mfu_avg = np.concatenate(all_mfu_avg)

    # Sort by step and deduplicate (keep last occurrence for overlapping steps)
    order = np.argsort(steps, kind='stable')
    steps, tps, tps_avg, mfu, mfu_avg = (
        steps[order], tps[order], tps_avg[order], mfu[order], mfu_avg[order]
    )
    _, unique_idx = np.unique(steps, return_index=True)
    steps = steps[unique_idx]
    tps = tps[unique_idx]
    tps_avg = tps_avg[unique_idx]
    mfu = mfu[unique_idx]
    mfu_avg = mfu_avg[unique_idx]

    return {
        'steps': steps,
        'tps': tps,
        'tps_avg': tps_avg,
        'mfu': mfu * MFU_CORRECTION,
        'mfu_avg': mfu_avg * MFU_CORRECTION,
    }


def parse_grep_metrics(log_path: Path) -> dict:
    """Parse grep-extracted metric lines (step header + TPS/MFU lines)."""
    text = log_path.read_text()
    steps, tps, tps_avg, mfu, mfu_avg = [], [], [], [], []

    block_pat = re.compile(r'\[step=(\d+)/\d+,epoch=\d+,eta=')
    tps_pat = re.compile(r'throughput/device/TPS=([\d,]+)')
    tps_avg_pat = re.compile(r'throughput/device/TPS \(actual avg\)=([\d,]+)')
    mfu_pat = re.compile(r'throughput/device/MFU=([\d.]+)')
    mfu_avg_pat = re.compile(r'throughput/device/MFU \(actual avg\)=([\d.]+)')

    current_step = None
    current = {}

    for line in text.split('\n'):
        bm = block_pat.search(line)
        if bm:
            if current_step is not None and 'tps' in current:
                steps.append(current_step)
                tps.append(current['tps'])
                tps_avg.append(current.get('tps_avg', current['tps']))
                mfu.append(current['mfu'])
                mfu_avg.append(current.get('mfu_avg', current['mfu']))
            current_step = int(bm.group(1))
            current = {}
            continue

        if current_step is not None:
            m = tps_pat.search(line)
            if m:
                current['tps'] = float(m.group(1).replace(',', ''))
            m = tps_avg_pat.search(line)
            if m:
                current['tps_avg'] = float(m.group(1).replace(',', ''))
            m = mfu_pat.search(line)
            if m:
                current['mfu'] = float(m.group(1))
            m = mfu_avg_pat.search(line)
            if m:
                current['mfu_avg'] = float(m.group(1))

    if current_step is not None and 'tps' in current:
        steps.append(current_step)
        tps.append(current['tps'])
        tps_avg.append(current.get('tps_avg', current['tps']))
        mfu.append(current['mfu'])
        mfu_avg.append(current.get('mfu_avg', current['mfu']))

    return {
        'steps': np.array(steps),
        'tps': np.array(tps),
        'tps_avg': np.array(tps_avg),
        'mfu': np.array(mfu) * MFU_CORRECTION,
        'mfu_avg': np.array(mfu_avg) * MFU_CORRECTION,
    }


def main():
    # Load data
    # kislurm v1 — Think was preempted many times, merge all wandb sessions
    ki_think = parse_wandb_multi(
        CKPT / "dolci-think-sft/wandb/wandb"
    )
    ki_instruct = parse_wandb_log(
        CKPT / "dolci-instruct-sft/wandb/wandb/run-20251229_191221-0c77cvcj/files/output.log"
    )

    # HoreKa v2 (grep-extracted)
    hk_think = parse_grep_metrics(ROOT / "logs/horeka_think_v2_metrics.txt")
    hk_instruct = parse_grep_metrics(ROOT / "logs/horeka_instruct_v2_metrics.txt")

    for name, data in [("kislurm Think", ki_think), ("kislurm Instruct", ki_instruct),
                        ("HoreKa Think", hk_think), ("HoreKa Instruct", hk_instruct)]:
        print(f"{name:20s}: {len(data['steps']):5d} datapoints, steps {data['steps'][0]}-{data['steps'][-1]}")

    # --- Plot style ---
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'legend.fontsize': 9, 'figure.dpi': 150,
    })
    colors = {
        'ki_think': '#2196F3', 'ki_instruct': '#FF9800',
        'hk_think': '#4CAF50', 'hk_instruct': '#E91E63',
    }
    labels = {
        'ki_think': 'Think (kislurm 1×8)',
        'ki_instruct': 'Instruct (kislurm 1×8)',
        'hk_think': 'Think (HoreKa 2×4)',
        'hk_instruct': 'Instruct (HoreKa 2×4)',
    }

    # ====== Figure 1: TPS/device over steps ======
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle('Tokens Per Second (TPS) per Device over Training', fontsize=14, fontweight='bold')

    # Think (left)
    for key, data in [('ki_think', ki_think), ('hk_think', hk_think)]:
        axes[0].scatter(data['steps'], data['tps'], alpha=0.15, s=4, color=colors[key])
        axes[0].plot(data['steps'], data['tps_avg'], color=colors[key], linewidth=2,
                     label=f"{labels[key]} (avg: {data['tps_avg'][-1]:,.0f})")
    axes[0].set_title('Think SFT')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('TPS / device')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Instruct (right)
    for key, data in [('ki_instruct', ki_instruct), ('hk_instruct', hk_instruct)]:
        axes[1].scatter(data['steps'], data['tps'], alpha=0.15, s=4, color=colors[key])
        axes[1].plot(data['steps'], data['tps_avg'], color=colors[key], linewidth=2,
                     label=f"{labels[key]} (avg: {data['tps_avg'][-1]:,.0f})")
    axes[1].set_title('Instruct SFT')
    axes[1].set_xlabel('Training Step')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    for out_dir in [FIGURES, EVAL_FIGURES]:
        fig.savefig(out_dir / 'throughput_tps.png', bbox_inches='tight')
    print(f"Saved throughput_tps.png")
    plt.close()

    # ====== Figure 2: MFU (H200-corrected) over steps ======
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle('Model FLOPs Utilization (MFU) over Training — H200 SXM BF16 basis', fontsize=14, fontweight='bold')

    for idx, (model, pairs) in enumerate([
        ('Think SFT', [('ki_think', ki_think), ('hk_think', hk_think)]),
        ('Instruct SFT', [('ki_instruct', ki_instruct), ('hk_instruct', hk_instruct)]),
    ]):
        ax = axes[idx]
        for key, data in pairs:
            ax.scatter(data['steps'], data['mfu'], alpha=0.15, s=4, color=colors[key])
            ax.plot(data['steps'], data['mfu_avg'], color=colors[key], linewidth=2,
                    label=f"{labels[key]} (avg: {data['mfu_avg'][-1]:.1f}%)")
        ax.set_title(model)
        ax.set_xlabel('Training Step')
        if idx == 0:
            ax.set_ylabel('MFU (%)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(50, 100)

    plt.tight_layout()
    for out_dir in [FIGURES, EVAL_FIGURES]:
        fig.savefig(out_dir / 'throughput_mfu.png', bbox_inches='tight')
    print(f"Saved throughput_mfu.png")
    plt.close()


if __name__ == '__main__':
    main()
