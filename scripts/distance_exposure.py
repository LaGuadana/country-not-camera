"""Separate distance from exposure, and test the multi-country explanation (CPU only).

For new trips of seen photographers (test_shared), the excess accuracy E (correct minus the same model's
test accuracy for that country) is tabulated jointly by
  exposure  = the photographer's number of training images (manifest train split), and
  distance  = km to the same photographer's nearest training image,
keeping only trips in a country where the photographer also has training images. A second table reports,
per exposure bin, the share of test_shared images in a country the photographer never drove in training
and the excess with and without those images. Photographer-clustered bootstrap CIs.

  python -I scripts/distance_exposure.py runs/ft_imagenet runs/ft_imagenet_fp --out results/final/distance_exposure.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from place_vs_camera import nearest_train_km  # noqa: E402

EXP = [(0, 5, "2-5"), (5, 12, "6-12"), (12, 25, "13-25"), (25, 10 ** 9, "26+")]
DIST = [(0, 25, "<25 km"), (25, 100, "25-100 km"), (100, 1e9, ">=100 km")]


def boot(b, rng, n=1000):
    if len(b) == 0:
        return None, None
    codes, inv = np.unique(b.creator_id, return_inverse=True)
    w = rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)), size=n)[:, inv]
    d = (w @ b.ex.to_numpy()) / w.sum(1)
    return float(b.ex.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = pd.read_csv("data/c32/manifest.csv", dtype=str, keep_default_na=False)
    tr = man[man.split == "train"]
    counts = tr.creator_id.value_counts()
    pc = set(zip(tr.creator_id, tr.country))
    dist = nearest_train_km(man)
    rng = np.random.default_rng(0)
    out = {"runs": {}}
    for run in a.runs:
        p = pd.read_csv(f"{run}/eval/predictions.csv", dtype=str, keep_default_na=False)
        p["correct"] = (p.pred == p.country).astype(float)
        acc_c = p[p.split == "test"].groupby("country").correct.mean()
        q = p[p.split == "test_shared"].copy()
        q["ex"] = q.correct - q.country.map(acc_c)
        q["n"] = q.creator_id.map(counts).fillna(0).astype(int)
        q["km"] = q.id.map(dist)
        q["same"] = [(c, k) in pc for c, k in zip(q.creator_id, q.country)]
        grid, multi = [], []
        print(run)
        for elo, ehi, elab in EXP:
            qe = q[(q.n > elo) & (q.n <= ehi)]
            row = {"exposure": elab}
            for dlo, dhi, dlab in DIST:
                b = qe[qe.same & (qe.km >= dlo) & (qe.km < dhi)]
                m, c = boot(b, rng)
                row[dlab] = {"images": int(len(b)), "excess": m, "ci95": c}
            grid.append(row)
            m_all, c_all = boot(qe, rng)
            m_same, c_same = boot(qe[qe.same], rng)
            multi.append({"exposure": elab, "images": int(len(qe)), "share_other_country": float((~qe.same).mean()),
                          "excess_all": m_all, "ci_all": c_all, "excess_same_country": m_same, "ci_same": c_same,
                          "excess_other_country": float(qe[~qe.same].ex.mean()) if (~qe.same).any() else None})
            cells = "  ".join(f"{d}: {100*row[d]['excess']:+5.1f} (n={row[d]['images']})" if row[d]["excess"] is not None else f"{d}: n/a"
                              for _, _, d in DIST)
            mm = multi[-1]
            print(f"  exposure {elab:5s} | {cells} | other-country share {100*mm['share_other_country']:4.1f}%  "
                  f"excess all {100*m_all:+5.1f} -> same-country only {100*m_same:+5.1f}")
        out["runs"][run] = {"exposure_x_distance": grid, "multi_country": multi}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
