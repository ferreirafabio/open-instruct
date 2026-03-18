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
    # Load v2 data only (production runs used for evaluation)
    #
    # Think v2: started on kislurm (steps 0-35K), checkpoint transferred to
    # HoreKa to finish (steps 36K-42,856). Merge all sources:
    # 1. kislurm wandb sessions (steps 10-35,273)
    # 2. kislurm SLURM log for the gap session (steps 35,260-36,518)
    # 3. HoreKa wandb sessions (steps 36,010-42,850)
    v2_think_local = parse_wandb_multi(
        CKPT / "dolci-think-sft-v2/wandb/wandb"
    )
    v2_think_gap = parse_grep_metrics(ROOT / "logs/kislurm_think_v2_gap_metrics.txt")
    v2_think_horeka = parse_grep_metrics(ROOT / "logs/horeka_think_v2_metrics.txt")
    # Merge gap data into local (both are kislurm)
    for k in v2_think_local:
        v2_think_local[k] = np.concatenate([v2_think_local[k], v2_think_gap[k]])
    order = np.argsort(v2_think_local['steps'], kind='stable')
    for k in v2_think_local:
        v2_think_local[k] = v2_think_local[k][order]
    _, unique_idx = np.unique(v2_think_local['steps'], return_index=True)
    for k in v2_think_local:
        v2_think_local[k] = v2_think_local[k][unique_idx]

    v2_think = {
        'steps': np.concatenate([v2_think_local['steps'], v2_think_horeka['steps']]),
        'tps': np.concatenate([v2_think_local['tps'], v2_think_horeka['tps']]),
        'tps_avg': np.concatenate([v2_think_local['tps_avg'], v2_think_horeka['tps_avg']]),
        'mfu': np.concatenate([v2_think_local['mfu'], v2_think_horeka['mfu']]),
        'mfu_avg': np.concatenate([v2_think_local['mfu_avg'], v2_think_horeka['mfu_avg']]),
    }
    order = np.argsort(v2_think['steps'], kind='stable')
    for k in v2_think:
        v2_think[k] = v2_think[k][order]
    _, unique_idx = np.unique(v2_think['steps'], return_index=True)
    for k in v2_think:
        v2_think[k] = v2_think[k][unique_idx]

    # Instruct v2: trained entirely on HoreKa
    v2_instruct = parse_grep_metrics(ROOT / "logs/horeka_instruct_v2_metrics.txt")

    for name, data in [("Think v2", v2_think), ("Instruct v2", v2_instruct)]:
        print(f"{name:20s}: {len(data['steps']):5d} datapoints, steps {data['steps'][0]}-{data['steps'][-1]}")

    # --- Plot style ---
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'legend.fontsize': 9, 'figure.dpi': 150,
    })

    def filter_outliers(steps, data, tps_key='tps', min_tps=6000):
        """Remove warmup/checkpoint-saving outliers (TPS < min_tps)."""
        mask = data[tps_key] >= min_tps
        return steps[mask], {k: v[mask] for k, v in data.items()}

    def rolling_stats(values, window=50):
        """Compute rolling median with 10th-90th percentile band.

        Uses adaptive window at edges to avoid padding artifacts.
        """
        n = len(values)
        if n < window:
            window = max(1, n // 3)
        half = window // 2
        median = np.empty(n)
        p10 = np.empty(n)
        p90 = np.empty(n)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            chunk = values[lo:hi]
            median[i] = np.median(chunk)
            p10[i] = np.percentile(chunk, 10)
            p90[i] = np.percentile(chunk, 90)
        return median, p10, p90

    # Split Think v2 into kislurm and HoreKa segments for color-coding.
    # Use HoreKa's first step as the clean cutoff — no overlap.
    horeka_start = v2_think_horeka['steps'][0] if len(v2_think_horeka['steps']) > 0 else 36010
    ki_mask = v2_think['steps'] < horeka_start
    hk_mask = v2_think['steps'] >= horeka_start

    think_ki = {k: v2_think[k][ki_mask] for k in v2_think}
    think_hk = {k: v2_think[k][hk_mask] for k in v2_think}

    COLOR_KI = '#2196F3'   # blue for kislurm
    COLOR_HK = '#4CAF50'   # green for HoreKa
    COLOR_INST = '#FF9800'  # orange for instruct

    def plot_think_segments(ax, metric_key, avg_key):
        """Plot Think with kislurm/HoreKa color distinction."""
        for seg, color, label in [
            (think_ki, COLOR_KI, f'kislurm 1×8 H200'),
            (think_hk, COLOR_HK, f'HoreKa 2×4 H200'),
        ]:
            if len(seg['steps']) == 0:
                continue
            steps_f, seg_f = filter_outliers(seg['steps'], seg)
            median, p10, p90 = rolling_stats(seg_f[metric_key])
            ax.plot(steps_f, median, color=color, linewidth=2, label=label)
            ax.fill_between(steps_f, p10, p90, color=color, alpha=0.2)

    # ====== Figure 1: TPS/device over steps ======
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Tokens Per Second (TPS) per Device over Training (8× H200 SXM, BF16)',
                 fontsize=14, fontweight='bold')

    # Think (left) — split by cluster
    plot_think_segments(axes[0], 'tps', 'tps_avg')
    axes[0].set_title(f'Think SFT (42,856 steps, avg: {v2_think["tps_avg"][-1]:,.0f} TPS)')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('TPS / device')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Instruct (right) — all HoreKa (larger window: fewer datapoints than Think)
    steps_f, inst_f = filter_outliers(v2_instruct['steps'], v2_instruct)
    median, p10, p90 = rolling_stats(inst_f['tps'], window=100)
    axes[1].plot(steps_f, median, color=COLOR_INST, linewidth=2,
                 label=f"HoreKa 2×4 H200")
    axes[1].fill_between(steps_f, p10, p90,
                         color=COLOR_INST, alpha=0.2)
    axes[1].set_title(f'Instruct SFT (3,252 steps, avg: {v2_instruct["tps_avg"][-1]:,.0f} TPS)')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('TPS / device')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    for out_dir in [FIGURES, EVAL_FIGURES]:
        fig.savefig(out_dir / 'throughput_tps.png', bbox_inches='tight')
    print(f"Saved throughput_tps.png")
    plt.close()

    # ====== Figure 2: MFU (H200-corrected) over steps ======
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model FLOPs Utilization (MFU) over Training (8× H200 SXM, BF16 dense = 989.5 TFLOPS)',
                 fontsize=14, fontweight='bold')

    # Think (left) — split by cluster
    plot_think_segments(axes[0], 'mfu', 'mfu_avg')
    axes[0].set_title(f'Think SFT (42,856 steps, avg: {v2_think["mfu_avg"][-1]:.1f}%)')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('MFU (%)')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(50, 100)

    # Instruct (right) — all HoreKa (larger window: fewer datapoints than Think)
    steps_f, inst_f = filter_outliers(v2_instruct['steps'], v2_instruct)
    median, p10, p90 = rolling_stats(inst_f['mfu'], window=100)
    axes[1].plot(steps_f, median, color=COLOR_INST, linewidth=2,
                 label=f"HoreKa 2×4 H200")
    axes[1].fill_between(steps_f, p10, p90,
                         color=COLOR_INST, alpha=0.2)
    axes[1].set_title(f'Instruct SFT (3,252 steps, avg: {v2_instruct["mfu_avg"][-1]:.1f}%)')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('MFU (%)')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(50, 100)

    plt.tight_layout()
    for out_dir in [FIGURES, EVAL_FIGURES]:
        fig.savefig(out_dir / 'throughput_mfu.png', bbox_inches='tight')
    print(f"Saved throughput_mfu.png")
    plt.close()


if __name__ == '__main__':
    main()
