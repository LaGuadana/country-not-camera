"""Final-report figures: (a) BA on test / test_shared with 95 % photographer-bootstrap CIs and
(b) the photographer gap, for the baselines and the end-to-end pair; training curves of the
end-to-end pair. Writes report/final/figures/fig_e2e.{pdf,png} and fig_curves.{pdf,png}."""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "report/final/figures"
C_TEST, C_SHARED = "#1f4e79", "#c0504d"
mpl.rcParams.update({"font.size": 7.6, "axes.titlesize": 8.2, "xtick.labelsize": 6.8, "ytick.labelsize": 7.0,
                     "legend.fontsize": 7.0, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
ROWS = [("probe", "eval", "linear probe"), ("cpu_ft", "eval", "partial FT"),
        ("stage_3", "eval", "partial FT\n+3 chunks"), ("stage_3", "eval_blank", "+3 chunks,\nblanked at test"),
        ("ft_imagenet", "eval", "end-to-end"), ("ft_imagenet", "eval_blank", "end-to-end,\nblanked at test"),
        ("ft_imagenet_fp", "eval", "end-to-end\n+FP removal")]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)


rows = [(lab, json.load(open(f"runs/{r}/{sub}/metrics.json"))) for r, sub, lab in ROWS
        if os.path.exists(f"runs/{r}/{sub}/metrics.json")]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.7, 1]})
x = np.arange(len(rows))
w = 0.38
for off, split, color, label in ((-w / 2, "test", C_TEST, "test: unseen photographers"),
                                 (w / 2, "test_shared", C_SHARED, "test$_{shared}$: seen photographers")):
    y = np.array([100 * m["splits"][split]["balanced_acc"] for _, m in rows])
    lo = np.array([100 * m["splits"][split]["balanced_acc_ci95"][0] for _, m in rows])
    hi = np.array([100 * m["splits"][split]["balanced_acc_ci95"][1] for _, m in rows])
    ax1.bar(x + off, y, w, color=color, label=label)
    ax1.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt="none", ecolor="0.15", lw=0.7, capsize=1.5)
ax1.axhline(100 / 32, color="grey", lw=0.8, ls=":")
SHORT = {"linear probe": "probe", "partial FT": "partial\nFT", "partial FT\n+3 chunks": "+3\nchunks",
         "+3 chunks,\nblanked at test": "+3 ch.\nblank", "end-to-end": "E2E",
         "end-to-end,\nblanked at test": "E2E\nblank", "end-to-end\n+FP removal": "E2E\n+FP"}
ax1.set_xticks(x, [SHORT.get(lab, lab) for lab, _ in rows], fontsize=6.0)
ax1.set_ylim(0, 50)
ax1.set_ylabel("balanced accuracy (%)")
ax1.set_title("(a) accuracy on unseen vs. seen photographers", loc="left")
ax1.legend(frameon=False, loc="upper left")
g = np.array([100 * m["gap_test_shared_minus_test"]["balanced_acc"] for _, m in rows])
glo = np.array([100 * m["gap_test_shared_minus_test"]["balanced_acc_ci95"][0] for _, m in rows])
ghi = np.array([100 * m["gap_test_shared_minus_test"]["balanced_acc_ci95"][1] for _, m in rows])
ax2.errorbar(g, x, xerr=[g - glo, ghi - g], fmt="o", color="0.25", ms=3.5, capsize=2, lw=1)
ax2.axvline(0, color="0.3", lw=0.8)
ax2.set_yticks(x, [lab.replace("\n", " ") for lab, _ in rows], fontsize=6.2)
ax2.invert_yaxis()
ax2.set_xlabel("gap Δ (pp)")
ax2.set_title("(b) photographer gap, 95% CI", loc="left")
fig.tight_layout(w_pad=1.2)
save(fig, "fig_e2e")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.2))
for r, lab, c in (("ft_imagenet", "standard", C_TEST), ("ft_imagenet_fp", "fingerprint removal", "#6fa8dc")):
    if not os.path.exists(f"runs/{r}/metrics.csv"):
        continue
    m = pd.read_csv(f"runs/{r}/metrics.csv")
    ax1.plot(m.epoch + 1, 100 * m.val_balanced_acc, color=c, lw=1.4, label=f"{lab}: val BA")
    ax1.plot(m.epoch + 1, 100 * m.train_top1, color=c, lw=0.9, ls="--", label=f"{lab}: train top-1")
    ax2.plot(m.epoch + 1, 100 * m.shared_minus_disjoint_bacc, color=c, lw=1.4, marker="o", ms=2.5, label=lab)
ax1.set_xlabel("epoch")
ax1.set_ylabel("accuracy (%)")
ax1.set_title("(a) end-to-end training", loc="left")
ax1.legend(frameon=False, fontsize=6.2)
ax2.axhline(0, color="0.3", lw=0.8)
ax2.set_xlabel("epoch")
ax2.set_ylabel("BA(val$_{shared}$) − BA(val) (pp)")
ax2.set_title("(b) photographer gap on validation", loc="left")
ax2.legend(frameon=False)
fig.tight_layout(w_pad=1.5)
save(fig, "fig_curves")
print("wrote", OUT)
