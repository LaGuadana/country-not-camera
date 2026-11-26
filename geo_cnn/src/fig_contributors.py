"""Contributor concentration in OSV-5M train: cumulative share and effective contributors per country.

  python -I src/fig_contributors.py results/train_country_creator_counts.csv \
      results/photographers_by_country.csv figures/fig_g_contributors
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

counts_csv, per_country_csv, out = sys.argv[1:4]
BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6})

c = pd.read_csv(counts_csv, dtype={"creator_id": str})
per = c.groupby("creator_id").n.sum().sort_values(ascending=False).values
cum = np.cumsum(per) / per.sum()
S = pd.read_csv(per_country_csv, index_col=0).dropna(subset=["train_images"])
S = S[S.train_images >= 10000]

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
r = np.arange(1, len(per) + 1)
ax[0].plot(r, 100 * cum, color=BLUE, lw=2)
for k in (10, 100, 1000):
    ax[0].plot(k, 100 * cum[k - 1], "o", color=BLUE, ms=4)
    ax[0].annotate(f"top {k}: {100 * cum[k - 1]:.0f}%", (k, 100 * cum[k - 1]), xytext=(6, -10),
                   textcoords="offset points", fontsize=8, color=INK)
ax[0].set_xscale("log"); ax[0].set_ylim(0, 102); ax[0].set_xlim(1, len(per))
ax[0].set_xlabel(f"contributors, ranked by images (n = {len(per):,})")
ax[0].set_ylabel("share of training images (%)")
ax[0].set_title("(a) Cumulative share of training images", loc="left", fontsize=9)

ax[1].scatter(S.train_images, S.train_eff_photographers, s=14, color=BLUE, alpha=0.8, lw=0)
for code in ["US", "DE", "FR", "ES", "RU", "AU", "JP", "PK", "MA", "IN", "SE", "IT"]:
    if code in S.index:
        off = (3, -9) if code in ("IT", "JP") else (3, 2)
        ax[1].annotate(code, (S.train_images[code], S.train_eff_photographers[code]),
                       xytext=off, textcoords="offset points", fontsize=7.5, color=INK2)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("training images per country")
ax[1].set_ylabel("effective contributors (1 / Σp²)")
ax[1].set_title(f"(b) Countries with ≥10k images (n = {len(S)})", loc="left", fontsize=9)
fig.tight_layout(w_pad=2)
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
