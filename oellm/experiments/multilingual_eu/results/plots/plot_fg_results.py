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
F_ELO_EN = [954, 947, 935, 912, 762]
F_ELO_WOEN = [695, 711, 700, 685, 680]
F_N = [500, 486, 453, 359, 234]  # in k

# G-track (matched 166k)
G_RATIOS = [100, 75, 50, 25, 0]
G_NAMES = ["G1-100en", "G2-75en", "G3-50en", "G4-25en", "G5-0en"]
G_ELO = [739, 751, 737, 722, None]  # G5 tba
G_ELO_EN = [987, 951, 914, 961, None]
G_ELO_WOEN = [702, 709, 708, 694, None]

# All models for scatter + heatmap
ALL_MODELS = {
    "Baseline":  {"elo": 741, "en": 950, "woen": 722, "track": "Baseline"},
    "A1-90en":   {"elo": 702, "en": 771, "woen": 692, "track": "A"},
    "A2-80en":   {"elo": 704, "en": 769, "woen": 703, "track": "A"},
    "A3-70en":   {"elo": 713, "en": 766, "woen": 689, "track": "A"},
    "B1-90en":   {"elo": 720, "en": 789, "woen": 708, "track": "B"},
    "B2-80en":   {"elo": 722, "en": 797, "woen": 722, "track": "B"},
    "C0-100en":  {"elo": 670, "en": 791, "woen": 681, "track": "C"},
    "D1-90en":   {"elo": 751, "en": 942, "woen": 716, "track": "D"},
    "D2-80en":   {"elo": 751, "en": 956, "woen": 725, "track": "D"},
    "D3-70en":   {"elo": 753, "en": 963, "woen": 731, "track": "D"},
    "E1-90en":   {"elo": 758, "en": 965, "woen": 740, "track": "E"},
    "E2-80en":   {"elo": 759, "en": 931, "woen": 725, "track": "E"},
    "E3-70en":   {"elo": 751, "en": 940, "woen": 726, "track": "E"},
    "F1-100en":  {"elo": 728, "en": 954, "woen": 695, "track": "F"},
    "F2-75en":   {"elo": 747, "en": 947, "woen": 711, "track": "F"},
    "F3-50en":   {"elo": 726, "en": 935, "woen": 700, "track": "F"},
    "F4-25en":   {"elo": 739, "en": 912, "woen": 685, "track": "F"},
    "F5-0en":    {"elo": 677, "en": 762, "woen": 680, "track": "F"},
    "G1-100en":  {"elo": 739, "en": 987, "woen": 702, "track": "G"},
    "G2-75en":   {"elo": 751, "en": 951, "woen": 709, "track": "G"},
    "G3-50en":   {"elo": 737, "en": 914, "woen": 708, "track": "G"},
    "G4-25en":   {"elo": 722, "en": 961, "woen": 694, "track": "G"},
}

# Per-language Elo for heatmap
LANGS = ["en", "de", "es", "fr", "it", "pt", "pl", "nl", "cs", "ro", "el", "uk"]
HEATMAP_MODELS = [
    "Baseline",
    "A1-90en", "A3-70en",
    "C0-100en",
    "D1-90en", "D3-70en",
    "E1-90en",
    "F1-100en", "F2-75en", "F3-50en", "F4-25en", "F5-0en",
    "G1-100en", "G2-75en", "G3-50en", "G4-25en", "G5-0en",
]
PER_LANG = {
    "Baseline":  [950, 647, 691, 675, 771, 704, 721, 749, 696, 863, 641, 664],
    "A1-90en":   [771, 613, 750, 612, 771, 650, 634, 714, 718, 726, 661, 677],
    "A3-70en":   [766, 515, 733, 693, 784, 706, 721, 706, 684, 845, 618, 634],
    "C0-100en":  [791, 612, 670, 654, 756, 641, 691, 689, 648, 705, 686, 650],
    "D1-90en":   [942, 619, 744, 672, 766, 651, 712, 718, 697, 888, 644, 687],
    "D3-70en":   [962, 643, 708, 663, 784, 706, 734, 723, 710, 845, 627, 664],
    "E1-90en":   [965, 642, 711, 695, 807, 712, 732, 758, 703, 754, 604, 634],
    "F1-100en":  [954, 614, 718, 664, 742, 680, 634, 714, 677, 892, 636, 671],
    "F2-75en":   [947, 624, 713, 702, 776, 697, 708, 708, 666, 854, 612, 624],
    "F3-50en":   [935, 658, 696, 712, 788, 635, 750, 679, 653, 825, 652, 637],
    "F4-25en":   [912, 601, 732, 646, 800, 638, 715, 649, 661, 784, 605, 689],
    "F5-0en":    [762, 607, 685, 619, 775, 601, 686, 641, 621, 718, 575, 564],
    "G1-100en":  [987, 549, 733, 685, 781, 700, 680, 740, 586, 819, 596, 623],
    "G2-75en":   [950, 685, 696, 689, 757, 700, 711, 781, 665, 878, 676, 675],
    "G3-50en":   [914, 631, 715, 707, 788, 710, 667, 649, 647, 841, 659, 655],
    "G4-25en":   [960, 623, 712, 676, 774, 651, 656, 682, 661, 826, 660, 656],
    "G5-0en":    [776, 640, 701, 696, 751, 628, 636, 646, 585, 833, 543, 650],
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

for ax, (metric, f_vals, g_vals, title) in zip(axes, [
    ("Elo (all langs)", F_ELO, G_ELO, "Overall Elo"),
    ("Elo (English only)", F_ELO_EN, G_ELO_EN, "English Elo"),
    ("Elo (w/o English)", F_ELO_WOEN, G_ELO_WOEN, "Non-English Elo"),
]):
    # F-track
    ax.plot(F_RATIOS, f_vals, "o--", color=TRACK_COLORS["F"], label="F-track (varying N)", markersize=7, linewidth=1.5)
    for i, (r, v, n) in enumerate(zip(F_RATIOS, f_vals, F_N)):
        ax.annotate(f"{n}k", (r, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=TRACK_COLORS["F"], alpha=0.7)

    # G-track (skip None)
    g_r = [r for r, v in zip(G_RATIOS, g_vals) if v is not None]
    g_v = [v for v in g_vals if v is not None]
    ax.plot(g_r, g_v, "s-", color=TRACK_COLORS["G"], label="G-track (166k matched)", markersize=7, linewidth=1.5)

    # Baseline reference
    baseline_val = {"Elo (all langs)": 741, "Elo (English only)": 950, "Elo (w/o English)": 722}[metric]
    ax.axhline(y=baseline_val, color=TRACK_COLORS["Baseline"], linestyle=":", linewidth=1, alpha=0.7, label="Baseline")

    ax.set_xlabel("English %")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(F_RATIOS)
    ax.invert_xaxis()
    ax.grid(alpha=0.2)

axes[0].legend(fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fg_ratio_curve.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR}/fg_ratio_curve.png")
plt.close()

# ============================================================
# Plot 2: English vs EU tradeoff scatter
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for name, d in ALL_MODELS.items():
    track = d["track"]
    color = TRACK_COLORS[track]
    marker = {"Baseline": "*", "A": "o", "B": "D", "C": "X", "D": "s", "E": "^", "F": "v", "G": "P"}[track]
    size = 120 if track == "Baseline" else 60
    ax.scatter(d["en"], d["woen"], c=color, marker=marker, s=size, zorder=3, edgecolors="white", linewidth=0.5)
    # Label select models
    if name in ["Baseline", "F5-0en", "E1-90en", "F2-75en", "G2-75en", "C0-100en", "A3-70en", "D1-90en"]:
        offset = {"Baseline": (8, 5), "F5-0en": (8, -5), "E1-90en": (8, 5), "F2-75en": (-45, 8),
                  "G2-75en": (8, -8), "C0-100en": (8, -5), "A3-70en": (-15, 8), "D1-90en": (-40, -10)}
        ax.annotate(name, (d["en"], d["woen"]), textcoords="offset points",
                    xytext=offset.get(name, (8, 5)), fontsize=8, color=color)

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
    Line2D([0], [0], marker="P", color="w", markerfacecolor=TRACK_COLORS["G"], markersize=8, label="G (matched 166k)"),
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
track_breaks = [1, 3, 4, 6, 7, 12, 17]  # after Baseline, A, C, D, E, F, G
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
