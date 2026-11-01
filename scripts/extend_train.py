"""More training images for an existing subset, in chunks, under the same contributor rules.

Recomputes the contributor split and the held-out val_shared/test_shared groups from train.csv
exactly as select_subset.py did (its arguments and seed, read from subset_stats.json), and
checks them against the existing manifest. Extra images come only from train contributors and
from outside the held-out groups. No contributor ends up with more than --max-per-contributor
training images per country, counting the ones already in the manifest. Evaluation splits are
unchanged.

The extra images are ordered as in water-filling: a contributor gets its next image (round-robin
over its sequences) only after every contributor of that country with fewer images has had theirs,
ties at random. They are then dealt round-robin into ceil(n / --chunk-size) chunks, so every chunk
has the same mix of countries and of contributors' first and later images.

Writes <data-dir>/train_extra.csv: the manifest columns plus `chunk` (1, 2, ...).

Usage:
  python -I scripts/extend_train.py --data-dir data/c32
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_subset import assign_contributors, hold_out_shared, link_groups, parse_budget, read_metadata  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-csv", default=None, help="default: the one recorded in subset_stats.json")
    ap.add_argument("--chunk-size", type=int, default=None, help="default: the size of the existing train split")
    ap.add_argument("--max-per-contributor", type=int, default=None, help="default: the subset's cap")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    stats = json.load(open(os.path.join(a.data_dir, "subset_stats.json")))
    s = stats["args"]
    budget = parse_budget(s["budget"])
    cap = a.max_per_contributor or s["max_per_contributor"]

    # same steps and random draws as select_subset.main, up to the per-split selection
    rng = np.random.default_rng(s["seed"])
    countries = [c.strip() for c in s["countries"].split(",") if c.strip()]
    df, chosen, _, _ = read_metadata(a.train_csv or s["train_csv"], s["min_contributors"], countries)
    fracs = [float(x) for x in s["contrib_split"].split(",")]
    df["csplit"] = df["creator_id"].map(assign_contributors(df, fracs, rng))
    tr = df[df["csplit"] == "train"].copy()
    tr["group"] = link_groups(tr, s["cell_deg"])
    p_val = budget["val_shared"] / max(1, budget["val_shared"] + budget["test_shared"])
    held = hold_out_shared(tr, s["shared_frac"], p_val, rng)
    df["pool"] = df["csplit"]
    df.loc[tr.index, "pool"] = tr["group"].map(held).fillna("train")
    del tr

    man = pd.read_csv(os.path.join(a.data_dir, "manifest.csv"), dtype={"id": str, "creator_id": str},
                      keep_default_na=False)
    pool = df.set_index("id")["pool"]
    wrong = (man["id"].astype(object).map(pool) != man["split"]).sum()
    if wrong:
        raise SystemExit(f"{wrong} manifest rows disagree with the recomputed split; not extending")
    print(f"recomputed split matches all {len(man):,} manifest rows", flush=True)

    cand = df[(df["pool"] == "train") & ~df["id"].astype(object).isin(set(man["id"]))].drop(columns=["csplit", "pool"])
    del df, pool
    have = man[man["split"] == "train"].groupby(["country", "creator_id"]).size().rename("have")
    rng = np.random.default_rng(a.seed)
    # round-robin over each contributor's sequences, as in select_subset.pick
    cand = cand.assign(u=rng.random(len(cand))).sort_values("u")
    cand["r_seq"] = cand.groupby(["country", "creator_id", "sequence"]).cumcount()
    seq_codes, seq_uniques = pd.factorize(cand["sequence"])
    cand["seq_key"] = rng.random(len(seq_uniques))[seq_codes]
    cand = cand.sort_values(["country", "creator_id", "r_seq", "seq_key"])
    cand = cand.join(have, on=["country", "creator_id"])
    cand["rank"] = cand.groupby(["country", "creator_id"]).cumcount() + cand["have"].fillna(0).astype(int)
    cand = cand[cand["rank"] < cap]
    # water-filling order inside a country (by rank, ties at random), dealt round-robin into chunks
    n_train = int((man["split"] == "train").sum())
    k = max(1, -(-len(cand) // (a.chunk_size or n_train)))
    cand = cand.assign(u=rng.random(len(cand))).sort_values(["country", "rank", "u"])
    start = pd.Series(rng.integers(0, k, cand["country"].nunique()), index=cand["country"].unique())
    cand["chunk"] = (cand.groupby("country").cumcount() + cand["country"].map(start)) % k + 1

    classes = json.load(open(os.path.join(a.data_dir, "classes.json")))
    cand["split"] = "train"
    cand["label"] = cand["country"].map({c: i for i, c in enumerate(classes)})
    cols = [c for c in man.columns if c in cand.columns] + ["chunk"]
    out = cand[cols].sort_values(["chunk", "country"])
    out.to_csv(os.path.join(a.data_dir, "train_extra.csv"), index=False)

    per = out.groupby(["chunk", "country"]).size().unstack(fill_value=0)
    print(f"{len(out):,} extra training images (cap {cap} per contributor and country) in {len(per)} chunks; "
          f"already in the manifest: {n_train:,}")
    for c, r in per.iterrows():
        print(f"  chunk {c}: {r.sum():,} images, {int((r > 0).sum())} countries, per country {r.min()}-{r.max()}")
    left = per.sum().sort_values()
    print("extra images per country:", ", ".join(f"{c} {n:,}" for c, n in left.items()))


if __name__ == "__main__":
    main()
