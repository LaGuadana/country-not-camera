"""Build the balanced 10-country subset and a leakage-aware train/test split.

- Take the K countries with the most images in the downloaded shard.
- Balance: sample the same number n per country (n = size of the K-th largest country,
  optionally capped by --max-per-class), seed-fixed.
- Grouping: images are linked if they share a Mapillary `sequence` OR fall in the same
  0.1 x 0.1 degree lat/lon cell (~11 km) [OR, with --group-creator, share the photographer
  `creator_id`]; connected components = groups. Groups never
  straddle train and test (StratifiedGroupKFold, one fold held out = ~20 %).

Usage:
  python -I src/make_split.py --index data/processed/shard_index.csv \
      --out data/processed/split.csv --results results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class DSU:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def link_by(dsu, keys):
    first = {}
    for i, k in enumerate(keys):
        if k in first:
            dsu.union(first[k], i)
        else:
            first[k] = i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = no cap")
    ap.add_argument("--cell-deg", type=float, default=0.1)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-creator", action="store_true",
                    help="also link images by the same photographer (creator_id): stricter split")
    ap.add_argument("--tag", default="", help="suffix for the stats file, e.g. _strict")
    a = ap.parse_args()

    df = pd.read_csv(a.index, dtype={"id": str, "country": str, "sequence": str, "creator_id": str},
                     keep_default_na=False)
    vc = df["country"].value_counts()
    top = vc.index[: a.k].tolist()
    n = int(vc.iloc[a.k - 1])
    if a.max_per_class:
        n = min(n, a.max_per_class)
    sub = (df[df["country"].isin(top)]
           .groupby("country")
           .sample(n=n, random_state=a.seed)
           .reset_index(drop=True))

    dsu = DSU(len(sub))
    link_by(dsu, sub["sequence"].tolist())
    cell = (np.floor(sub["latitude"] / a.cell_deg).astype(int).astype(str) + "_" +
            np.floor(sub["longitude"] / a.cell_deg).astype(int).astype(str))
    link_by(dsu, cell.tolist())
    if a.group_creator:
        link_by(dsu, sub["creator_id"].tolist())
    sub["group"] = [dsu.find(i) for i in range(len(sub))]

    sgkf = StratifiedGroupKFold(n_splits=a.n_splits, shuffle=True, random_state=a.seed)
    tr_idx, te_idx = next(sgkf.split(sub, sub["country"], sub["group"]))
    sub["split"] = "train"
    sub.loc[te_idx, "split"] = "test"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sub.to_csv(a.out, index=False)

    g_tr = set(sub.loc[sub.split == "train", "group"])
    g_te = set(sub.loc[sub.split == "test", "group"])
    s_tr = set(sub.loc[sub.split == "train", "sequence"])
    s_te = set(sub.loc[sub.split == "test", "sequence"])
    c_tr = set(sub.loc[sub.split == "train", "creator_id"])
    c_te = set(sub.loc[sub.split == "test", "creator_id"])
    te = sub[sub.split == "test"]
    stats = {
        "countries": top,
        "shard_counts_of_chosen": {c: int(vc[c]) for c in top},
        "n_per_class": n,
        "n_total": len(sub),
        "n_train": int((sub.split == "train").sum()),
        "n_test": int((sub.split == "test").sum()),
        "per_class_train": sub[sub.split == "train"]["country"].value_counts().to_dict(),
        "per_class_test": te["country"].value_counts().to_dict(),
        "n_sequences": int(sub["sequence"].nunique()),
        "n_groups": int(sub["group"].nunique()),
        "largest_group": int(sub["group"].value_counts().iloc[0]),
        "groups_shared_train_test": len(g_tr & g_te),
        "sequences_shared_train_test": len(s_tr & s_te),
        "creators_total": len(c_tr | c_te),
        "creators_shared_train_test": len(c_tr & c_te),
        "test_images_whose_creator_in_train": int(te["creator_id"].isin(c_tr).sum()),
        "cell_deg": a.cell_deg,
        "group_by_creator": bool(a.group_creator),
        "seed": a.seed,
    }
    os.makedirs(a.results, exist_ok=True)
    with open(os.path.join(a.results, f"split_stats{a.tag}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
