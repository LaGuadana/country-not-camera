"""Restore data/c32 on a new machine with the ORIGINAL evaluation splits.

select_subset.py is not bit-reproducible across platforms: in hold_out_shared() the "largest group"
of a contributor is found with an unstable sort (pandas quicksort), and 494 contributors tie for their
largest group. x86 SIMD argsort and the Apple-ARM fallback break those ties differently, which changes
the held-out groups, the number of RNG draws, and from there every later water-filling draw. The
contributor -> train/val/test assignment does not depend on it and is reproduced exactly.

This script therefore:
  1. recomputes the contributor split (identical to the original; checked against the 38,118
     evaluation images listed in the committed runs/<run>/eval/predictions.csv),
  2. takes val, test, val_shared and test_shared verbatim from those predictions,
  3. links training-contributor images into groups (same Mapillary sequence or same contributor and
     0.1 deg cell, as in select_subset.py) and drops every group that contains an original
     val_shared/test_shared image, so no held-out trip can leak into training,
  4. water-fills a fresh training split from the remaining images with the original budget
     (3,200 per country, <= 100 per contributor).
The evaluation images are exactly those of the progress report; the training split (used by the
end-to-end runs runs/ft_imagenet*) is a statistically equivalent re-draw. The resulting
data/c32/manifest.csv is committed, so this script only needs to be re-run to rebuild it.

Usage:
  python -I scripts/restore_manifest.py --train-csv ../geo_cnn/data/raw/train.csv \
      --predictions runs/stage_3/eval/predictions.csv --out-dir data/c32
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_subset import SPLITS, assign_contributors, eff_contributors, link_groups, pick, read_metadata  # noqa: E402

COLS = ["id", "split", "label", "country", "creator_id", "sequence", "latitude", "longitude",
        "captured_at", "drive_side", "climate"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--predictions", required=True, help="committed predictions.csv listing the original eval images")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-budget", type=int, default=3200)
    ap.add_argument("--max-per-contributor", type=int, default=100)
    ap.add_argument("--cell-deg", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df, chosen, per_country, read_stats = read_metadata(a.train_csv, 200, [])
    rng = np.random.default_rng(a.seed)  # same first draws as select_subset.py
    df["csplit"] = df["creator_id"].map(assign_contributors(df, [0.7, 0.1, 0.2], rng))

    ev = pd.read_csv(a.predictions, dtype=str, keep_default_na=False, usecols=["id", "split", "creator_id", "country"])
    meta = df.set_index("id")
    missing = set(ev["id"]) - set(meta.index)
    if missing:
        raise SystemExit(f"{len(missing)} original eval ids not in train.csv metadata")
    # contributor split check: disjoint eval contributors are val/test, shared ones are train
    cs = meta.loc[ev["id"], "csplit"].to_numpy()
    want = ev["split"].map({"val": "val", "test": "test", "val_shared": "train", "test_shared": "train"}).to_numpy()
    n_bad = int((cs != want).sum())
    if n_bad:
        raise SystemExit(f"contributor split differs from the original for {n_bad} eval images")
    if not (meta.loc[ev["id"], "creator_id"].to_numpy() == ev["creator_id"].to_numpy()).all():
        raise SystemExit("creator ids differ from the predictions file")

    tr = df[df["csplit"] == "train"].copy()
    tr["group"] = link_groups(tr, a.cell_deg)
    shared_ids = set(ev.loc[ev["split"].isin(["val_shared", "test_shared"]), "id"])
    excl = set(tr.loc[tr["id"].isin(shared_ids), "group"])
    pool = tr[~tr["group"].isin(excl)].drop(columns=["group"])
    picked = pick(pool, a.train_budget, a.max_per_contributor, np.random.default_rng(a.seed + 1000))

    classes = sorted(chosen)
    lab = {c: i for i, c in enumerate(classes)}
    evrows = meta.loc[ev["id"]].reset_index()
    evrows["split"] = ev["split"].to_numpy()
    man = pd.concat([picked.assign(split="train"), evrows], ignore_index=True)
    man["label"] = man["country"].map(lab)
    man = man[COLS]

    by = {s: man[man["split"] == s] for s in SPLITS}
    cre = {s: set(by[s]["creator_id"]) for s in SPLITS}
    seq = {s: set(by[s]["sequence"]) for s in SPLITS}
    gid = dict(zip(tr["id"], tr["group"]))
    train_groups = {gid[i] for i in by["train"]["id"]}
    shared_groups = {gid[i] for s in ("val_shared", "test_shared") for i in by[s]["id"]}
    checks = {
        "contributors train & (val|test)": len(cre["train"] & (cre["val"] | cre["test"])),
        "contributors val & test": len(cre["val"] & cre["test"]),
        "sequences train & any eval split": len(seq["train"] & set().union(*(seq[s] for s in SPLITS[1:]))),
        "groups train & (val_shared|test_shared)": len(train_groups & shared_groups),
        "duplicate ids": int(man["id"].duplicated().sum()),
    }
    if any(checks.values()):
        raise SystemExit(f"leakage check failed: {checks}")
    not_seen = {s: int((~by[s]["creator_id"].isin(cre["train"])).sum()) for s in ("val_shared", "test_shared")}

    stats = {"method": "restore_manifest.py: original eval splits + fresh water-filled train split",
             "args": vars(a), "read": read_stats, "classes": classes, "checks": checks,
             "shared_images_whose_contributor_has_no_train_image": not_seen,
             "train_pool_images": int(len(pool)), "excluded_groups": len(excl), "splits": {}}
    for s in SPLITS:
        d = by[s]
        per_c = d.groupby("creator_id").size()
        stats["splits"][s] = {
            "images": len(d), "contributors": int(d["creator_id"].nunique()),
            "sequences": int(d["sequence"].nunique()),
            "images_per_contributor_median_max": [float(per_c.median()), int(per_c.max())],
            "per_country": {c: {"images": int(len(g)), "contributors": int(g["creator_id"].nunique()),
                                "eff_contributors": round(eff_contributors(g["creator_id"]), 1)}
                            for c, g in d.groupby("country")}}
    os.makedirs(a.out_dir, exist_ok=True)
    man.to_csv(os.path.join(a.out_dir, "manifest.csv"), index=False)
    with open(os.path.join(a.out_dir, "classes.json"), "w") as f:
        json.dump(classes, f)
    with open(os.path.join(a.out_dir, "subset_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    for s in SPLITS:
        st = stats["splits"][s]
        imgs = [v["images"] for v in st["per_country"].values()]
        print(f"{s:12s} {st['images']:7,d} images  {st['contributors']:6,d} contributors  "
              f"per country {min(imgs)}-{max(imgs)}")
    print("checks:", checks, "| shared images without a train image of their contributor:", not_seen)


if __name__ == "__main__":
    main()
