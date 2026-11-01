"""Decompose the contributor gap by how many training images the contributor had.

The test_shared split holds out whole trips of *training* contributors. If the model
memorises contributor-specific cues (camera, mount, colour profile), its accuracy on a
contributor's held-out trips should grow with that contributor's number of training
images. If the gap were only geography (seen contributors photograph typical places),
accuracy would be flat in that count. Contributors of the disjoint test split have zero
training images and serve as the reference level.

Binning: 0 (test), then quantiles of the training-count distribution among test_shared
contributors, fixed across runs. Accuracy is plain top-1 within a bin (each country has
400 test / 400 test_shared images, so bins stay country-balanced to first order); a
contributor-clustered bootstrap gives 95 % CIs.

Usage:
  python -I scripts/shortcut_decomposition.py runs/probe runs/cpu_ft runs/stage_3
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from geocnn.metrics import ci  # noqa: E402

EDGES = [1, 6, 21, 51, 10 ** 9]  # 1-5, 6-20, 21-50, >50 train images


def contributor_train_counts(data_dir):
    man = pd.read_csv(os.path.join(data_dir, "manifest.csv"),
                      dtype={"id": str, "creator_id": str}, keep_default_na=False)
    extra = pd.read_csv(os.path.join(data_dir, "train_extra.csv"),
                        dtype={"id": str, "creator_id": str}, keep_default_na=False)
    n_man = man[man["split"] == "train"]["creator_id"].value_counts()
    n_ext = extra["creator_id"].value_counts()
    return (n_man.add(n_ext, fill_value=0)).astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data-dir", default="data/c32")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    counts = contributor_train_counts(a.data_dir)
    # bin edges from the distribution of train counts among test_shared contributors
    shared_pred = pd.read_csv(os.path.join(a.runs[0], "eval", "predictions.csv"),
                              dtype=str, keep_default_na=False)
    shared_creators = shared_pred[shared_pred["split"] == "test_shared"]["creator_id"].unique()
    cs = counts.reindex(shared_creators).dropna()
    print(f"test_shared contributors: {len(cs):,}; train images each: "
          f"median {cs.median():.0f}, p90 {cs.quantile(0.9):.0f}, max {cs.max():.0f}")

    out = {"bins": ["0 (test)", "1", "2-6", "7-21", "22-51", "52+"], "runs": {}}
    for run in a.runs:
        p = pd.read_csv(os.path.join(run, "eval", "predictions.csv"),
                        dtype=str, keep_default_na=False)
        p["correct"] = (p["pred"] == p["country"]).astype(float)
        p["n_train"] = p["creator_id"].map(counts).fillna(0).astype(int)
        # country-adjusted excess: bin accuracy minus the same countries' disjoint-test accuracy,
        # weighted by each country's share of the bin (kills the country-mix confound)
        acc_test_c = p[p["split"] == "test"].groupby("country")["correct"].mean()

        def excess(b):
            per_c = b.groupby("country")["correct"].agg(["mean", "size"])
            per_c = per_c.join(acc_test_c.rename("acc_test"))
            return float((per_c["mean"] - per_c["acc_test"]).mul(per_c["size"]).sum()
                         / per_c["size"].sum())
        rows = []
        rng = np.random.default_rng(a.seed)
        for split in ("test", "test_shared"):
            q = p[p["split"] == split]
            bins = [(-1, 10 ** 9, "0 (test)")] if split == "test" else \
                list(zip([0] + EDGES, EDGES, ["1", "2-6", "7-21", "22-51", "52+"]))
            for lo, hi, lab in bins:
                b = q[(q["n_train"] > lo) & (q["n_train"] <= hi)]
                if b.empty:
                    rows.append({"bin": lab, "split": split, "n": 0, "top1": None, "ci95": None})
                    continue
                # contributor-clustered bootstrap of the bin's top-1 and of the
                # country-adjusted excess (per-image correct minus the image country's
                # disjoint-test accuracy; the test-side estimate is treated as fixed)
                codes, inv = np.unique(b["creator_id"], return_inverse=True)
                v = (b["correct"] - b["country"].map(acc_test_c)).to_numpy()
                draws, ex_draws = np.empty(a.n_boot), np.empty(a.n_boot)
                for s in range(0, a.n_boot, 100):
                    w = rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)),
                                        size=min(100, a.n_boot - s))
                    ww = w[:, inv]
                    draws[s:s + 100] = (ww @ b["correct"].to_numpy()) / ww.sum(1)
                    ex_draws[s:s + 100] = (ww @ v) / ww.sum(1)
                rows.append({"bin": lab, "split": split,
                             "n": int(len(b)), "contributors": int(len(codes)),
                             "top1": float(b["correct"].mean()), "ci95": ci(draws),
                             "excess_vs_test_same_countries": excess(b) if split == "test_shared" else 0.0,
                             "excess_ci95": ci(ex_draws) if split == "test_shared" else None})
        out["runs"][os.path.basename(run)] = rows
        lab = {r["bin"]: f"{r['top1']:.3f}" if r["top1"] is not None else "-" for r in rows}
        print(f"{os.path.basename(run):10s} " +
              "  ".join(f"{b}:{lab.get(b, '-')}" for b in out["bins"]), flush=True)
    path = os.path.join(os.path.dirname(a.runs[0].rstrip("/")) or ".", "shortcut_bins.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
