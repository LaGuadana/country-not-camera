"""Figures for the progress report.

(a) balanced accuracy on the two test protocols as labelled data grows (with 95 %
    contributor-clustered CIs): the two lines diverge, i.e. the shortcut grows;
(b) the gap decomposed by the contributor's number of training images: country-adjusted
    excess accuracy of held-out trips, flat for frozen features, rising with exposure for
    fine-tuned models.

Reads the runs' eval/metrics.json and runs/shortcut_bins.json; writes PDF + PNG next to
report/progress/figures/.
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

RUNS = [("probe", "linear probe (frozen)", 9600),
        ("cpu_ft", "fine-tune", 101578),
        ("stage_1", "+chunk 1", 188768),
        ("stage_2", "+chunk 2", 275962),
        ("stage_3", "+chunk 3", 363150)]
C_TEST, C_SHARED = "#1f4e79", "#c0504d"

mpl.rcParams.update({"font.size": 7.6, "axes.titlesize": 8.2, "axes.labelsize": 7.6,
                     "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.45))

# ---- (a) scaling on the two protocols ----------------------------------------
x = np.array([n for _, _, n in RUNS], float)
for split, color, label in (("test", C_TEST, "test: unseen contrib."),
                            ("test_shared", C_SHARED, "test$_{shared}$: seen contrib.")):
    y = np.array([json.load(open(f"runs/{r}/eval/metrics.json"))["splits"][split]["balanced_acc"]
                  for r, _, _ in RUNS])
    lo = np.array([json.load(open(f"runs/{r}/eval/metrics.json"))["splits"][split]["balanced_acc_ci95"][0]
                   for r, _, _ in RUNS])
    hi = np.array([json.load(open(f"runs/{r}/eval/metrics.json"))["splits"][split]["balanced_acc_ci95"][1]
                   for r, _, _ in RUNS])
    ax1.errorbar(x, 100 * y, yerr=[100 * (y - lo), 100 * (hi - y)], color=color, marker="o",
                 ms=3, lw=1.4, capsize=2, label=label)
ax1.axhline(100 / 32, color="grey", lw=0.8, ls=":")
ax1.text(11000, 100 / 32 + 1.2, "chance 3.1", color="grey", fontsize=6.6)
ax1.set_xscale("log")
ax1.set_xticks(x, ["probe", "fine-tune", "+1", "+2", "+3"], rotation=0)
ax1.set_xlabel("distinct training images (log scale)")
ax1.set_ylabel("balanced accuracy (%)")
ax1.set_title("(a) more data, wider contributor gap", loc="left")
ax1.set_ylim(0, 45)
ax1.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.10), handlelength=1.4)
gap = 100 * (json.load(open("runs/stage_3/eval/metrics.json"))
             ["gap_test_shared_minus_test"]["balanced_acc"])
ax1.annotate(f"gap {gap:.1f} pp", xy=(x[-1], 100 * 0.345), xytext=(1.4e5, 100 * 0.235),
             fontsize=7, arrowprops=dict(arrowstyle="-", color="0.35", lw=0.8), color="0.15")

# ---- (b) excess accuracy by contributor exposure ------------------------------
bins = json.load(open("runs/shortcut_bins.json"))["bins"][2:]  # drop "0 (test)" and the empty "1"
labels = ["2–6", "7–21", "22–51", "≥52"]
pos = np.arange(len(labels))
for run, color, label in (("probe", "#7f7f7f", "linear probe (frozen)"),
                          ("cpu_ft", "#4f81bd", "fine-tune"),
                          ("stage_3", C_SHARED, "fine-tune + 3 chunks")):
    rows = {r["bin"]: r for r in json.load(open("runs/shortcut_bins.json"))["runs"][run]
            if r["split"] == "test_shared"}
    y = np.array([100 * rows[b]["excess_vs_test_same_countries"] for b in bins])
    lo = np.array([100 * rows[b]["excess_ci95"][0] for b in bins])
    hi = np.array([100 * rows[b]["excess_ci95"][1] for b in bins])
    ax2.errorbar(pos, y, yerr=[y - lo, hi - y], color=color,
                 marker="o", ms=3, lw=1.4, capsize=2, label=label)
ax2.axhline(0, color="0.2", lw=0.8)
ax2.set_xticks(pos, labels)
ax2.set_xlabel("training images of that contributor")
ax2.set_ylabel("excess accuracy vs. test (pp)")
ax2.set_title("(b) shortcut grows with contributor exposure", loc="left")
ax2.legend(frameon=False, loc="lower right", handlelength=1.4)

os.makedirs("report/progress/figures", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"report/progress/figures/scaling_decomposition.{ext}")
print("wrote report/progress/figures/scaling_decomposition.{pdf,png}")
