"""Generate F/G-track visualizations for taskboard issue post.

Creates three plots:
1. Ratio curve: English % vs Elo (overall, en, w/o en)
2. English vs EU tradeoff scatter
3. Per-language Elo heatmap

Usage:
    python oellm/experiments/multilingual_eu/results/plots/plot_fg_results.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["font.family"] = "sans-serif"

OUTDIR = "oellm/experiments/multilingual_eu/results/plots"

# ============================================================
# Data
# ============================================================

# F-track (unmatched samples)
F_RATIOS = [100, 75, 50, 25, 0]
F_NAMES = ["F1-100en", "F2-75en", "F3-50en", "F4-25en", "F5-0en"]
F_ELO = [728, 747, 726, 739, 677]
F_ELO_CI = [9, 10, 9, 9, 11]
F_ELO_EN = [954, 947, 935, 912, 762]
F_ELO_EN_CI = [22, 22, 21, 22, 31]
F_ELO_WOEN = [695, 711, 700, 685, 680]
F_ELO_WOEN_CI = [11, 10, 12, 11, 11]
F_N = [500, 486, 453, 359, 234]  # in k

# G-track (matched 166k)
G_RATIOS = [100, 75, 50, 25, 0]
G_NAMES = ["G1-100en", "G2-75en", "G3-50en", "G4-25en", "G5-0en"]
G_ELO = [739, 751, 737, 722, None]  # G5 tba
G_ELO_EN = [987, 951, 914, 961, None]
G_ELO_WOEN = [702, 709, 708, 694, None]

# All models for scatter + heatmap
ALL_MODELS = {
    "Baseline":  {"elo": 741, "en": 950, "woen": 722, "en_ci": 21, "woen_ci": 10, "track": "Baseline"},
    "A1-90en":   {"elo": 702, "en": 771, "woen": 692, "en_ci": 32, "woen_ci": 10, "track": "A"},
    "A2-80en":   {"elo": 704, "en": 769, "woen": 703, "en_ci": 30, "woen_ci": 11, "track": "A"},
    "A3-70en":   {"elo": 713, "en": 766, "woen": 689, "en_ci": 29, "woen_ci": 11, "track": "A"},
    "B1-90en":   {"elo": 720, "en": 789, "woen": 708, "en_ci": 26, "woen_ci": 10, "track": "B"},
    "B2-80en":   {"elo": 722, "en": 797, "woen": 722, "en_ci": 26, "woen_ci": 10, "track": "B"},
    "C0-100en":  {"elo": 670, "en": 791, "woen": 681, "en_ci": 29, "woen_ci": 11, "track": "C"},
    "D1-90en":   {"elo": 751, "en": 942, "woen": 716, "en_ci": 20, "woen_ci": 11, "track": "D"},
    "D2-80en":   {"elo": 751, "en": 956, "woen": 725, "en_ci": 20, "woen_ci": 12, "track": "D"},
    "D3-70en":   {"elo": 753, "en": 963, "woen": 731, "en_ci": 21, "woen_ci": 11, "track": "D"},
    "E1-90en":   {"elo": 758, "en": 965, "woen": 740, "en_ci": 21, "woen_ci": 9, "track": "E"},
    "E2-80en":   {"elo": 759, "en": 931, "woen": 725, "en_ci": 24, "woen_ci": 9, "track": "E"},
    "E3-70en":   {"elo": 751, "en": 940, "woen": 726, "en_ci": 22, "woen_ci": 9, "track": "E"},
    "F1-100en":  {"elo": 728, "en": 954, "woen": 695, "en_ci": 22, "woen_ci": 11, "track": "F"},
    "F2-75en":   {"elo": 747, "en": 947, "woen": 711, "en_ci": 22, "woen_ci": 10, "track": "F"},
    "F3-50en":   {"elo": 726, "en": 935, "woen": 700, "en_ci": 21, "woen_ci": 12, "track": "F"},
    "F4-25en":   {"elo": 739, "en": 912, "woen": 685, "en_ci": 22, "woen_ci": 11, "track": "F"},
    "F5-0en":    {"elo": 677, "en": 762, "woen": 680, "en_ci": 31, "woen_ci": 11, "track": "F"},
}

# Per-language Elo for heatmap
LANGS = ["en", "de", "es", "fr", "it", "pt", "pl", "nl", "cs", "ro", "el", "uk"]
HEATMAP_MODELS = [
    "Baseline",
    "A1-90en", "A2-80en", "A3-70en",
    "B1-90en", "B2-80en",
    "C0-100en",
    "D1-90en", "D2-80en", "D3-70en",
    "E1-90en", "E2-80en", "E3-70en",
    "F1-100en", "F2-75en", "F3-50en", "F4-25en", "F5-0en",
]
PER_LANG = {
    "Baseline":  [950, 647, 691, 675, 771, 704, 721, 749, 696, 863, 641, 664],
    "A1-90en":   [771, 613, 750, 612, 771, 650, 634, 714, 718, 726, 661, 677],
    "A2-80en":   [769, 574, 735, 663, 779, 671, 732, 712, 675, 688, 640, 666],
    "A3-70en":   [766, 515, 733, 693, 784, 706, 721, 706, 684, 845, 618, 634],
    "B1-90en":   [789, 615, 692, 675, 776, 705, 742, 717, 732, 875, 595, 620],
    "B2-80en":   [797, 668, 698, 711, 807, 729, 740, 690, 676, 815, 662, 652],
    "C0-100en":  [791, 612, 670, 654, 756, 641, 691, 689, 648, 705, 686, 650],
    "D1-90en":   [942, 619, 744, 672, 766, 651, 712, 718, 697, 888, 644, 687],
    "D2-80en":   [956, 589, 708, 699, 746, 683, 663, 742, 706, 854, 688, 656],
    "D3-70en":   [962, 643, 708, 663, 784, 706, 734, 723, 710, 845, 627, 664],
    "E1-90en":   [965, 642, 711, 695, 807, 712, 732, 758, 703, 754, 604, 634],
    "E2-80en":   [931, 681, 726, 608, 751, 645, 705, 754, 706, 838, 673, 634],
    "E3-70en":   [940, 613, 721, 682, 781, 676, 676, 757, 694, 844, 658, 642],
    "F1-100en":  [954, 614, 718, 664, 742, 680, 634, 714, 677, 892, 636, 671],
    "F2-75en":   [947, 624, 713, 702, 776, 697, 708, 708, 666, 854, 612, 624],
    "F3-50en":   [935, 658, 696, 712, 788, 635, 750, 679, 653, 825, 652, 637],
    "F4-25en":   [912, 601, 732, 646, 800, 638, 715, 649, 661, 784, 605, 689],
    "F5-0en":    [762, 607, 685, 619, 775, 601, 686, 641, 621, 718, 575, 564],
}

TRACK_COLORS = {
    "Baseline": "#2c2c2c",
    "A": "#e74c3c",
    "B": "#e67e22",
    "C": "#95a5a6",
    "D": "#3498db",
    "E": "#2ecc71",
    "F": "#9b59b6",
    "G": "#f39c12",
}

# ============================================================
# Plot 1: Ratio curve
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

for ax, (metric, f_vals, f_ci, title) in zip(axes, [
    ("Elo (all langs)", F_ELO, F_ELO_CI, "Overall Elo"),
    ("Elo (English only)", F_ELO_EN, F_ELO_EN_CI, "English Elo"),
    ("Elo (w/o English)", F_ELO_WOEN, F_ELO_WOEN_CI, "Non-English Elo"),
]):
    f_vals_arr = np.array(f_vals)
    f_ci_arr = np.array(f_ci)
    # F-track with CI bands
    ax.fill_between(F_RATIOS, f_vals_arr - f_ci_arr, f_vals_arr + f_ci_arr,
                    color=TRACK_COLORS["F"], alpha=0.15)
    ax.plot(F_RATIOS, f_vals, "o-", color=TRACK_COLORS["F"], markersize=8, linewidth=2)

    # Baseline reference
    baseline_val = {"Elo (all langs)": 741, "Elo (English only)": 950, "Elo (w/o English)": 722}[metric]
    ax.axhline(y=baseline_val, color=TRACK_COLORS["Baseline"], linestyle=":", linewidth=1, alpha=0.7, label="Baseline")

    ax.set_xlabel("English %")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(F_RATIOS)
    ax.invert_xaxis()
    ax.grid(alpha=0.2)

axes[0].legend(fontsize=9, loc="lower left", framealpha=0.9)
fig.tight_layout()
fig.suptitle("Elo vs English/EU ratio (Dolci replay, ~500k samples)", fontsize=13, y=1.02)
fig.savefig(f"{OUTDIR}/fg_ratio_curve.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR}/fg_ratio_curve.png")
plt.close()

# ============================================================
# Plot 2: English vs EU tradeoff scatter
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

from adjustText import adjust_text
texts = []

for name, d in ALL_MODELS.items():
    track = d["track"]
    color = TRACK_COLORS[track]
    marker = {"Baseline": "*", "A": "o", "B": "D", "C": "X", "D": "s", "E": "^", "F": "v"}[track]
    size = 150 if track == "Baseline" else 70
    ax.errorbar(d["en"], d["woen"], xerr=d["en_ci"], yerr=d["woen_ci"],
                fmt="none", ecolor=color, alpha=0.3, elinewidth=1, capsize=2, zorder=2)
    ax.scatter(d["en"], d["woen"], c=color, marker=marker, s=size, zorder=3, edgecolors="white", linewidth=0.5)
    texts.append(ax.text(d["en"], d["woen"], name, fontsize=8, color=color))

adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5))

# Legend for tracks
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor=TRACK_COLORS["Baseline"], markersize=12, label="Baseline"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=TRACK_COLORS["A"], markersize=8, label="A (fusion-synth)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=TRACK_COLORS["B"], markersize=8, label="B (scaled)"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor=TRACK_COLORS["C"], markersize=8, label="C (English ctrl)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=TRACK_COLORS["D"], markersize=8, label="D (Dolci replay)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=TRACK_COLORS["E"], markersize=8, label="E (Dolci scaled)"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor=TRACK_COLORS["F"], markersize=8, label="F (extreme ratios)"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

ax.set_xlabel("Elo (English only)")
ax.set_ylabel("Elo (w/o English)")
ax.set_title("English vs EU tradeoff")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fg_tradeoff_scatter.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR}/fg_tradeoff_scatter.png")
plt.close()

# ============================================================
# Plot 3: Per-language Elo heatmap
# ============================================================

models = [m for m in HEATMAP_MODELS if m in PER_LANG]
data = np.array([PER_LANG[m] for m in models])

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=500, vmax=1000)

ax.set_xticks(range(len(LANGS)))
ax.set_xticklabels(LANGS, fontsize=10)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([f"  {m}" for m in models], fontsize=10)

# Add value annotations
for i in range(len(models)):
    for j in range(len(LANGS)):
        val = data[i, j]
        color = "white" if val < 650 or val > 900 else "black"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7.5, color=color)

# Add track separators
track_breaks = [1, 4, 6, 7, 10, 13]  # after Baseline, A, B, C, D, E
for b in track_breaks:
    if b < len(models):
        ax.axhline(y=b - 0.5, color="white", linewidth=2)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Elo rating", fontsize=10)

ax.set_title("Per-language Elo ratings (Qwen3.5-27B judge, 200 battles/lang)", fontsize=12)
ax.set_xlabel("Language")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fg_per_language_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR}/fg_per_language_heatmap.png")
plt.close()

print("All plots generated.")
