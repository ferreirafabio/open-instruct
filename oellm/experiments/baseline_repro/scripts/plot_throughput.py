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

# MFU correction: OLMo-core used A100 peak (156 TFLOPS) instead of H200 (989.5 TFLOPS)
A100_PEAK = 156e12
H200_PEAK = 989.5e12
MFU_CORRECTION = A100_PEAK / H200_PEAK


def parse_wandb_log(log_path: Path) -> dict:
    """Parse output.log for per-step metrics (logged every 10 steps)."""
    text = log_path.read_text()
    steps, tps, tps_avg, mfu, mfu_avg = [], [], [], [], []
    timestamps = []

    # Match blocks: step header followed by metric lines
    block_pat = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+\t.*\[step=(\d+)/\d+,epoch=\d+,eta=.*?\]'
    )
    tps_pat = re.compile(r'throughput/device/TPS=([\d,]+)')
    tps_avg_pat = re.compile(r'throughput/device/TPS \(actual avg\)=([\d,]+)')
    mfu_pat = re.compile(r'throughput/device/MFU=([\d.]+)')
    mfu_avg_pat = re.compile(r'throughput/device/MFU \(actual avg\)=([\d.]+)')

    lines = text.split('\n')
    current_step = None
    current_ts = None
    current = {}

    for line in lines:
        bm = block_pat.search(line)
        if bm:
            # Save previous block
            if current_step is not None and 'tps' in current:
                steps.append(current_step)
                timestamps.append(current_ts)
                tps.append(current['tps'])
                tps_avg.append(current.get('tps_avg', current['tps']))
                mfu.append(current['mfu'])
                mfu_avg.append(current.get('mfu_avg', current['mfu']))
            current_step = int(bm.group(2))
            current_ts = bm.group(1)
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

    # Save last block
    if current_step is not None and 'tps' in current:
        steps.append(current_step)
        timestamps.append(current_ts)
        tps.append(current['tps'])
        tps_avg.append(current.get('tps_avg', current['tps']))
        mfu.append(current['mfu'])
        mfu_avg.append(current.get('mfu_avg', current['mfu']))

    return {
        'steps': np.array(steps),
        'tps': np.array(tps),
        'tps_avg': np.array(tps_avg),
        'mfu': np.array(mfu) * MFU_CORRECTION,  # Correct to H200 basis
        'mfu_avg': np.array(mfu_avg) * MFU_CORRECTION,
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


def compute_wall_clock(steps: np.ndarray, bps_avg: float) -> np.ndarray:
    """Compute cumulative wall-clock hours from step numbers and avg BPS."""
    step_time = 1.0 / bps_avg  # seconds per step
    return steps * step_time / 3600


def main():
    # Load data
    # kislurm v1
    ki_think = parse_wandb_log(
        CKPT / "dolci-think-sft/wandb/wandb/run-20251228_004416-opfnhcpn/files/output.log"
    )
    ki_instruct = parse_wandb_log(
        CKPT / "dolci-instruct-sft/wandb/wandb/run-20251229_191221-0c77cvcj/files/output.log"
    )

    # HoreKa v2 (grep-extracted)
    hk_think = parse_grep_metrics(ROOT / "logs/horeka_think_v2_metrics.txt")
    hk_instruct = parse_grep_metrics(ROOT / "logs/horeka_instruct_v2_metrics.txt")

    print(f"kislurm Think:    {len(ki_think['steps'])} datapoints, steps {ki_think['steps'][0]}-{ki_think['steps'][-1]}")
    print(f"kislurm Instruct: {len(ki_instruct['steps'])} datapoints, steps {ki_instruct['steps'][0]}-{ki_instruct['steps'][-1]}")
    print(f"HoreKa Think:     {len(hk_think['steps'])} datapoints, steps {hk_think['steps'][0]}-{hk_think['steps'][-1]}")
    print(f"HoreKa Instruct:  {len(hk_instruct['steps'])} datapoints, steps {hk_instruct['steps'][0]}-{hk_instruct['steps'][-1]}")

    # BPS averages (from wandb-summary.json)
    bps = {
        'ki_think': 0.0594, 'ki_instruct': 0.0665,
        'hk_think': 0.05878, 'hk_instruct': 0.06661,
    }

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

    datasets = [
        ('ki_think', ki_think), ('ki_instruct', ki_instruct),
        ('hk_think', hk_think), ('hk_instruct', hk_instruct),
    ]

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
    fig.savefig(FIGURES / 'throughput_tps.png', bbox_inches='tight')
    print(f"Saved {FIGURES / 'throughput_tps.png'}")
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
    fig.savefig(FIGURES / 'throughput_mfu.png', bbox_inches='tight')
    print(f"Saved {FIGURES / 'throughput_mfu.png'}")
    plt.close()

    # ====== Figure 3: Cumulative wall-clock time ======
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cumulative Wall-Clock Time over Training', fontsize=14, fontweight='bold')

    # Think
    for key, data in [('ki_think', ki_think), ('hk_think', hk_think)]:
        wall = compute_wall_clock(data['steps'], bps[key])
        axes[0].plot(data['steps'], wall, color=colors[key], linewidth=2, label=labels[key])
    axes[0].set_title('Think SFT')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Wall-Clock Time (hours)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Instruct
    for key, data in [('ki_instruct', ki_instruct), ('hk_instruct', hk_instruct)]:
        wall = compute_wall_clock(data['steps'], bps[key])
        axes[1].plot(data['steps'], wall, color=colors[key], linewidth=2, label=labels[key])
    axes[1].set_title('Instruct SFT')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Wall-Clock Time (hours)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / 'throughput_wallclock.png', bbox_inches='tight')
    print(f"Saved {FIGURES / 'throughput_wallclock.png'}")
    plt.close()

    # ====== Figure 4: Combined overview (2x2) ======
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OLMo-3-7B SFT Training Throughput', fontsize=15, fontweight='bold', y=0.98)

    # Row 0: TPS
    for idx, (model, pairs) in enumerate([
        ('Think SFT', [('ki_think', ki_think), ('hk_think', hk_think)]),
        ('Instruct SFT', [('ki_instruct', ki_instruct), ('hk_instruct', hk_instruct)]),
    ]):
        ax = axes[0, idx]
        for key, data in pairs:
            ax.scatter(data['steps'], data['tps'], alpha=0.1, s=3, color=colors[key])
            ax.plot(data['steps'], data['tps_avg'], color=colors[key], linewidth=1.5,
                    label=f"{labels[key]} ({data['tps_avg'][-1]:,.0f})")
        ax.set_title(model)
        ax.set_ylabel('TPS / device')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    # Row 1: MFU
    for idx, (model, pairs) in enumerate([
        ('Think SFT', [('ki_think', ki_think), ('hk_think', hk_think)]),
        ('Instruct SFT', [('ki_instruct', ki_instruct), ('hk_instruct', hk_instruct)]),
    ]):
        ax = axes[1, idx]
        for key, data in pairs:
            ax.scatter(data['steps'], data['mfu'], alpha=0.1, s=3, color=colors[key])
            ax.plot(data['steps'], data['mfu_avg'], color=colors[key], linewidth=1.5,
                    label=f"{labels[key]} ({data['mfu_avg'][-1]:.1f}%)")
        ax.set_xlabel('Training Step')
        ax.set_ylabel('MFU (%) — H200 basis')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(50, 100)

    plt.tight_layout()
    fig.savefig(FIGURES / 'throughput_overview.png', bbox_inches='tight')
    print(f"Saved {FIGURES / 'throughput_overview.png'}")
    plt.close()


if __name__ == '__main__':
    main()
