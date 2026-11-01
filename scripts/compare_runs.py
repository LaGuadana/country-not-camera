"""Paired comparison of evaluated runs against a baseline on the same test images.

Both runs are scored on the same contributor-bootstrap draws (same clusters, same seed), so the
intervals are for the change itself, which is much tighter than comparing two separate intervals.

  $PY scripts/compare_runs.py runs/cpu_ft runs/stage_1 runs/stage_2 runs/stage_3
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from geocnn.metrics import ci, cluster_bootstrap  # noqa: E402


def load(run, splits):
    p = pd.read_csv(os.path.join(run, "eval", "predictions.csv"), dtype=str, keep_default_na=False)
    return p[p["split"].isin(splits)].set_index("id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data-dir", default="data/c32")
    ap.add_argument("--disjoint", default="test")
    ap.add_argument("--shared", default="test_shared")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    with open(os.path.join(a.data_dir, "classes.json")) as f:
        classes = json.load(f)
    label = {c: i for i, c in enumerate(classes)}
    base = load(a.baseline, [a.disjoint, a.shared])

    def bacc_draws(p, split):
        m = base["split"] == split
        correct = (p["pred"][m] == p["country"][m]).to_numpy(np.float64)
        return cluster_bootstrap(correct, base["country"][m].map(label).to_numpy(),
                                 base["creator_id"][m].to_numpy(), len(classes), a.n_boot, a.seed)[1]

    b = {s: bacc_draws(base, s) for s in (a.disjoint, a.shared)}
    print(f"change vs {a.baseline} (balanced accuracy, paired contributor bootstrap, 95% CI)")
    for run in a.runs:
        p = load(run, [a.disjoint, a.shared]).loc[base.index]
        r = {s: bacc_draws(p, s) for s in (a.disjoint, a.shared)}
        acc = r[a.disjoint] - b[a.disjoint]
        gap = (r[a.shared] - r[a.disjoint]) - (b[a.shared] - b[a.disjoint])
        lo, hi = ci(acc)
        glo, ghi = ci(gap)
        print(f"{run}: {a.disjoint} {np.mean(acc):+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"gap {np.mean(gap):+.3f} [{glo:+.3f}, {ghi:+.3f}]  P(gap change <= 0) {np.mean(gap <= 0):.3f}")


if __name__ == "__main__":
    main()
