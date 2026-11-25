"""Extra final-report analyses from predictions.csv files (no model runs).

  per-country recall on test / test_shared and the per-country gap, for several runs
  exposure dose-response of the end-to-end runs (training images per contributor in the
    manifest train split, which is what those runs saw), country-adjusted, contributor bootstrap
  structure of errors: same driving side, same world region, top confusions, vs. chance
Writes results/final/extra.json and report/final/figures/fig_countries.{pdf,png}, fig_exposure.{pdf,png}.
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = {"classical": "runs/classical/eval", "probe": "runs/probe/eval", "stage_3": "runs/stage_3/eval",
        "ft_imagenet": "runs/ft_imagenet/eval", "ft_imagenet_fp": "runs/ft_imagenet_fp/eval"}
LEFT = {"AU", "GB", "ID", "IN", "JP", "MY", "TH"}  # real-world left-hand traffic (OSV-5M marks GB as right)
REGION = {**{c: "Europe" for c in "AT BE CH CZ DE DK ES FR GB GR HR HU IT NL PL PT SE RU TR".split()},
          **{c: "Americas" for c in "BR CA CO MX US".split()},
          **{c: "Asia" for c in "ID IN JP MY TH TW".split()}, "AU": "Oceania", "ET": "Africa"}
rng = np.random.default_rng(0)
out = {}


def load(run):
    p = pd.read_csv(f"{RUNS[run]}/predictions.csv", dtype=str, keep_default_na=False)
    p["correct"] = (p["pred"] == p["country"]).astype(float)
    return p


P = {r: load(r) for r in RUNS if os.path.exists(f"{RUNS[r]}/predictions.csv")}
classes = sorted(P["stage_3"]["country"].unique())

# ---- per-country recall and gap ------------------------------------------------------------
pc = {}
for r in ("stage_3", "ft_imagenet", "ft_imagenet_fp"):
    p = P[r]
    t = p[p.split == "test"].groupby("country").correct.mean()
    s = p[p.split == "test_shared"].groupby("country").correct.mean()
    pc[r] = pd.DataFrame({"test": t, "test_shared": s, "gap": s - t})
gS, gF = pc["ft_imagenet"]["gap"], pc["ft_imagenet_fp"]["gap"]
red = (gS - gF).sort_values(ascending=False)
out["per_country"] = {r: d.round(4).to_dict() for r, d in pc.items()}
out["per_country_summary"] = {
    "corr_gap_standard_vs_fp": float(np.corrcoef(gS, gF)[0, 1]),
    "countries_gap_reduced": int((red > 0).sum()),
    "largest_gap_reductions": red.head(6).round(3).to_dict(),
    "largest_gap_increases": red.tail(4).round(3).to_dict(),
    "test_recall_change_fp": (pc["ft_imagenet_fp"]["test"] - pc["ft_imagenet"]["test"]).sort_values().round(3).to_dict(),
    "corr_gap_reduction_vs_standard_gap": float(np.corrcoef(gS, gS - gF)[0, 1]),
    "e2e_vs_stage3_test_change": (pc["ft_imagenet"]["test"] - pc["stage_3"]["test"]).sort_values().round(3).to_dict(),
}

# ---- exposure dose-response (end-to-end runs; manifest train counts) -------------------------
man = pd.read_csv("data/c32/manifest.csv", dtype=str, keep_default_na=False)
counts = man[man.split == "train"].creator_id.value_counts()
BINS = [(1, 5, "2-5"), (5, 12, "6-12"), (12, 25, "13-25"), (25, 10 ** 9, "26+")]
exp = {}
for r in ("ft_imagenet", "ft_imagenet_fp", "stage_3"):
    p = P[r]
    acc_c = p[p.split == "test"].groupby("country").correct.mean()
    q = p[p.split == "test_shared"].copy()
    q["n"] = q.creator_id.map(counts).fillna(0).astype(int)
    q["ex"] = q.correct - q.country.map(acc_c)
    rows = []
    for lo, hi, lab in BINS:
        b = q[(q.n > lo) & (q.n <= hi)]
        codes, inv = np.unique(b.creator_id, return_inverse=True)
        w = rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)), size=1000)[:, inv]
        d = (w @ b.ex.to_numpy()) / w.sum(1)
        rows.append({"bin": lab, "n": int(len(b)), "contributors": int(len(codes)),
                     "excess": float(b.ex.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]})
    exp[r] = rows
out["exposure_manifest_counts"] = exp
out["exposure_note"] = ("bins by the contributor's number of images in the manifest train split (what the end-to-end "
                        "runs saw); stage_3 saw more (extra chunks), so its row is binned on a count it did not use")

# ---- structure of errors ---------------------------------------------------------------------
err = {}
for r, p in P.items():
    t = p[(p.split == "test") & (p.correct == 0)]
    same_side = ((t.country.isin(LEFT)) == (t.pred.isin(LEFT))).mean()
    same_reg = (t.country.map(REGION) == t.pred.map(REGION)).mean()
    # chance: wrong label uniform over the 31 other classes
    ch_side = np.mean([(sum((c in LEFT) == (o in LEFT) for o in classes if o != c)) / 31 for c in t.country])
    ch_reg = np.mean([(sum(REGION[c] == REGION[o] for o in classes if o != c)) / 31 for c in t.country])
    left = p[(p.split == "test") & p.country.isin(LEFT)]
    err[r] = {"error_rate": float(1 - p[p.split == "test"].correct.mean()),
              "errors_same_drive_side": float(same_side), "chance_same_drive_side": float(ch_side),
              "errors_same_region": float(same_reg), "chance_same_region": float(ch_reg),
              "left_traffic_images_predicted_left": float(left.pred.isin(LEFT).mean())}
    if r in ("stage_3", "ft_imagenet", "ft_imagenet_fp"):
        conf = t.groupby(["country", "pred"]).size().sort_values(ascending=False)
        err[r]["top_confusions"] = {f"{a}->{b}": int(n) for (a, b), n in conf.head(10).items()}
out["errors"] = err
os.makedirs("results/final", exist_ok=True)
json.dump(out, open("results/final/extra.json", "w"), indent=1)

# ---- figures ---------------------------------------------------------------------------------
mpl.rcParams.update({"font.size": 7.6, "axes.titlesize": 8.2, "xtick.labelsize": 6.8, "ytick.labelsize": 7.0,
                     "legend.fontsize": 6.8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
OUT = "report/final/figures"
order = pc["ft_imagenet"]["test"].sort_values().index
x = np.arange(len(order))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True)
a1.scatter(x, 100 * pc["stage_3"]["test"][order], marker="x", s=14, color="0.5", label="partial FT +3 chunks (test)")
a1.vlines(x, 100 * pc["ft_imagenet"]["test"][order], 100 * pc["ft_imagenet"]["test_shared"][order], color="#1f4e79", lw=0.8, alpha=0.6)
a1.scatter(x, 100 * pc["ft_imagenet"]["test"][order], s=14, color="#1f4e79", label="end-to-end, test (unseen)")
a1.scatter(x, 100 * pc["ft_imagenet"]["test_shared"][order], s=14, facecolors="none", edgecolors="#c0504d",
           label="end-to-end, test$_{shared}$ (seen)")
a1.axhline(100 / 32, color="grey", lw=0.7, ls=":")
a1.set_ylabel("recall (%)")
a1.set_title("(a) recall per country, sorted by the end-to-end model on unseen photographers", loc="left")
a1.legend(frameon=False, ncol=3, loc="upper left")
w = 0.38
a2.bar(x - w / 2, 100 * gS[order], w, color="#1f4e79", label="end-to-end, standard")
a2.bar(x + w / 2, 100 * gF[order], w, color="#6fa8dc", label="end-to-end, fingerprint removal")
a2.axhline(0, color="0.3", lw=0.7)
a2.set_ylabel("gap: test$_{shared}$ − test (pp)")
a2.set_title("(b) photographer gap per country", loc="left")
a2.legend(frameon=False, ncol=2, loc="upper left")
a2.set_xticks(x, order)
fig.tight_layout(h_pad=1.0)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_countries.{ext}")
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.6, 2.3))
pos = np.arange(len(BINS))
for r, lab, c, off in (("ft_imagenet", "end-to-end, standard", "#1f4e79", -0.08),
                       ("ft_imagenet_fp", "end-to-end, FP removal", "#6fa8dc", 0.08)):
    y = np.array([100 * b["excess"] for b in exp[r]])
    lo = np.array([100 * b["ci95"][0] for b in exp[r]])
    hi = np.array([100 * b["ci95"][1] for b in exp[r]])
    ax.errorbar(pos + off, y, yerr=[y - lo, hi - y], color=c, marker="o", ms=3, lw=1.3, capsize=2, label=lab)
ax.axhline(0, color="0.3", lw=0.7)
ax.set_xticks(pos, [b[2] for b in BINS])
ax.set_xlabel("training images of that photographer")
ax.set_ylabel("excess accuracy vs. test (pp)")
ax.legend(frameon=False)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_exposure.{ext}")
plt.close(fig)
print(json.dumps({k: out[k] for k in ("per_country_summary", "exposure_manifest_counts", "errors")}, indent=1))
