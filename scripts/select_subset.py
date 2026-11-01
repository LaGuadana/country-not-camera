"""Select the contributor-capped, contributor-disjoint OSV-5M training subset.

1. Countries: those with >= --min-contributors distinct contributors (creator_id) in the OSV-5M
   train split (32 countries at 200).
2. Contributors -> train / val / test (default 70/10/20 % of contributors), stratified by each
   contributor's main country. A contributor belongs to exactly one of these, in every country.
3. Inside the train contributors, whole groups of images linked by the same Mapillary sequence or
   the same contributor-specific 0.1 deg cell (~11 km) are held out as val_shared / test_shared:
   trips and places the model never saw, taken by photographers it did see. Each contributor keeps
   its largest group in train. The accuracy gap test_shared - test is the contributor shortcut.
4. Per (split, country) a fixed image budget is filled by water-filling: every contributor gets at
   most c images, c being the largest cap that fits the budget, so the budget is spread over as many
   contributors as possible. Within a contributor, images are taken round-robin over its sequences.

Writes <out-dir>/manifest.csv, classes.json, subset_stats.json.

Usage:
  python -I scripts/select_subset.py --train-csv ~/ML/geo_cnn/data/raw/train.csv --out-dir data/c32
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

STR_COLS = ["id", "country", "sequence", "captured_at", "creator_id", "drive_side", "climate"]
SPLITS = ["train", "val", "test", "val_shared", "test_shared"]


def parse_budget(s):
    out = dict(kv.split("=") for kv in s.split(","))
    out = {k: int(v) for k, v in out.items()}
    missing = set(SPLITS) - set(out)
    if missing:
        raise SystemExit(f"--budget is missing {sorted(missing)}")
    return out


def read_metadata(path, min_contributors, countries):
    t0 = time.time()
    opts = pacsv.ConvertOptions(
        include_columns=STR_COLS + ["latitude", "longitude"],
        column_types={**{c: pa.string() for c in STR_COLS},
                      "latitude": pa.float64(), "longitude": pa.float64()},
        strings_can_be_null=False)
    tab = pacsv.read_csv(path, convert_options=opts)
    n_all = tab.num_rows
    # creator_id is written as a float ("111336221091714.0")
    tab = tab.set_column(tab.schema.get_field_index("creator_id"), "creator_id",
                         pc.replace_substring_regex(tab["creator_id"], r"\.0$", ""))
    no_creator = pc.equal(tab["creator_id"], "")
    n_no_creator = pc.sum(no_creator).as_py() or 0
    tab = tab.filter(pc.invert(no_creator))
    per_country = tab.group_by("country").aggregate([("creator_id", "count_distinct")]).to_pandas()
    per_country = per_country.set_index("country")["creator_id_count_distinct"].sort_values(ascending=False)
    if countries:
        chosen = sorted(countries)
    else:
        chosen = sorted(per_country[per_country >= min_contributors].index)
    tab = tab.filter(pc.is_in(tab["country"], value_set=pa.array(chosen)))
    df = tab.to_pandas()
    print(f"read {n_all:,} rows in {time.time() - t0:.0f}s; dropped {n_no_creator:,} without creator_id; "
          f"{len(chosen)} countries -> {len(df):,} rows", flush=True)
    return df, chosen, per_country, {"rows_total": n_all, "rows_without_creator": n_no_creator}


def assign_contributors(df, fracs, rng):
    """Each contributor -> train/val/test, stratified by its main country."""
    cnt = df.groupby(["creator_id", "country"]).size().rename("n").reset_index()
    main = cnt.sort_values(["creator_id", "n"], ascending=[True, False]).drop_duplicates("creator_id")
    main = main.assign(u=rng.random(len(main)))
    rank = main.groupby("country")["u"].rank(method="first") - 1
    r = rank / main.groupby("country")["u"].transform("size")
    split = np.where(r < fracs[0], "train", np.where(r < fracs[0] + fracs[1], "val", "test"))
    return dict(zip(main["creator_id"], split))


def link_groups(df, cell_deg):
    """Connected components of images linked by sequence or by (contributor, 0.1 deg cell)."""
    seq = pd.factorize(df["sequence"])[0]
    cy = np.floor(df["latitude"].to_numpy() / cell_deg).astype(np.int64)
    cx = np.floor(df["longitude"].to_numpy() / cell_deg).astype(np.int64)
    cell = pd.factorize(df["creator_id"].astype(str) + "|" + pd.Series(cy, index=df.index).astype(str)
                        + "|" + pd.Series(cx, index=df.index).astype(str))[0]
    n_seq, n_cell = seq.max() + 1, cell.max() + 1
    n = n_seq + n_cell
    g = coo_matrix((np.ones(len(df), dtype=np.int8), (seq, n_seq + cell)), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    return comp[seq]


def hold_out_shared(tr, frac_max, p_val, rng):
    """Pick whole groups of train contributors as val_shared / test_shared."""
    sizes = tr.groupby(["creator_id", "group"]).size().rename("n").reset_index()
    sizes = sizes.assign(u=rng.random(len(sizes)))
    held = {}
    for creator, g in sizes.groupby("creator_id", sort=False):
        if len(g) < 2:
            continue
        budget = frac_max * g["n"].sum()
        g = g.sort_values("n", ascending=False).iloc[1:].sort_values("u")  # never the largest group
        taken = 0
        for grp, n in zip(g["group"], g["n"]):
            if taken + n <= budget:
                held[grp] = "val_shared" if rng.random() < p_val else "test_shared"
                taken += n
    return held


def waterfill(n, budget, max_cap, rng):
    """Images per contributor: min(n_i, c) for the largest c that fits, then +1 for random
    contributors with more images until the budget is used."""
    n = np.minimum(n, max_cap)
    if n.sum() <= budget:
        return n
    lo, hi = 0, int(n.max())
    while lo < hi:
        c = (lo + hi + 1) // 2
        if np.minimum(n, c).sum() <= budget:
            lo = c
        else:
            hi = c - 1
    k = np.minimum(n, lo)
    rest = budget - int(k.sum())
    if rest > 0:
        cand = np.flatnonzero(n > lo)
        k[rng.choice(cand, size=rest, replace=False)] += 1
    return k


def pick(rows, budget, max_cap, rng):
    """Apply per-country water-filling and round-robin-over-sequences selection to one split."""
    rows = rows.assign(u=rng.random(len(rows))).sort_values("u")
    rows["r_seq"] = rows.groupby(["country", "creator_id", "sequence"]).cumcount()
    seq_codes, seq_uniques = pd.factorize(rows["sequence"])
    rows["seq_key"] = rng.random(len(seq_uniques))[seq_codes]
    rows = rows.sort_values(["country", "creator_id", "r_seq", "seq_key"])
    rows["rank"] = rows.groupby(["country", "creator_id"]).cumcount()
    keep = []
    for country, g in rows.groupby("country", sort=True):
        n = g.groupby("creator_id").size()
        k = pd.Series(waterfill(n.to_numpy(), budget, max_cap, rng), index=n.index)
        keep.append(g[g["rank"] < g["creator_id"].map(k)])
    return pd.concat(keep).drop(columns=["u", "r_seq", "seq_key", "rank"])


def eff_contributors(s):
    p = s.value_counts(normalize=True).to_numpy()
    return float(1.0 / np.sum(p ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", required=True, help="OSV-5M train.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-contributors", type=int, default=200)
    ap.add_argument("--countries", default="", help="comma-separated ISO codes; overrides --min-contributors")
    ap.add_argument("--contrib-split", default="0.7,0.1,0.2", help="train,val,test fractions of contributors")
    ap.add_argument("--shared-frac", type=float, default=0.15,
                    help="max share of a train contributor's images held out as val_shared/test_shared")
    ap.add_argument("--cell-deg", type=float, default=0.1)
    ap.add_argument("--budget", default="train=3200,val=200,test=400,val_shared=200,test_shared=400",
                    help="images per country and split")
    ap.add_argument("--max-per-contributor", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    fracs = [float(x) for x in a.contrib_split.split(",")]
    budget = parse_budget(a.budget)
    countries = [c.strip() for c in a.countries.split(",") if c.strip()]
    df, chosen, per_country, read_stats = read_metadata(a.train_csv, a.min_contributors, countries)

    df["csplit"] = df["creator_id"].map(assign_contributors(df, fracs, rng))
    tr = df[df["csplit"] == "train"].copy()
    tr["group"] = link_groups(tr, a.cell_deg)
    p_val = budget["val_shared"] / max(1, budget["val_shared"] + budget["test_shared"])
    held = hold_out_shared(tr, a.shared_frac, p_val, rng)
    df["pool"] = df["csplit"]
    df.loc[tr.index, "pool"] = tr["group"].map(held).fillna("train")

    picked = {s: pick(df[df["pool"] == s], budget[s], a.max_per_contributor, rng) for s in SPLITS}
    # "shared" means the contributor has images in the *selected* train set
    train_creators = set(picked["train"]["creator_id"])
    dropped_shared = {}
    for s in ("val_shared", "test_shared"):
        ok = picked[s]["creator_id"].isin(train_creators)
        dropped_shared[s] = int((~ok).sum())
        picked[s] = picked[s][ok]
    man = pd.concat([p.assign(split=s) for s, p in picked.items()], ignore_index=True)
    classes = sorted(chosen)
    man["label"] = man["country"].map({c: i for i, c in enumerate(classes)})
    man = man[["id", "split", "label", "country", "creator_id", "sequence", "latitude", "longitude",
               "captured_at", "drive_side", "climate"]]

    # leakage checks
    by = {s: man[man["split"] == s] for s in SPLITS}
    cre = {s: set(by[s]["creator_id"]) for s in SPLITS}
    seq = {s: set(by[s]["sequence"]) for s in SPLITS}
    checks = {
        "contributors train & (val|test)": len(cre["train"] & (cre["val"] | cre["test"])),
        "contributors val & test": len(cre["val"] & cre["test"]),
        "contributors (val_shared|test_shared) not in train": len((cre["val_shared"] | cre["test_shared"]) - cre["train"]),
        "sequences train & any eval split": len(seq["train"] & set().union(*(seq[s] for s in SPLITS[1:]))),
        "duplicate ids": int(man["id"].duplicated().sum()),
    }
    bad = {k: v for k, v in checks.items() if k != "contributors (val_shared|test_shared) not in train" and v}
    if bad or checks["contributors (val_shared|test_shared) not in train"]:
        raise SystemExit(f"leakage check failed: {checks}")

    stats = {"args": vars(a), "read": read_stats, "classes": classes,
             "contributors_per_country_full_train": {c: int(per_country[c]) for c in classes},
             "checks": checks, "shared_rows_dropped_contributor_not_in_selected_train": dropped_shared,
             "splits": {}}
    for s in SPLITS:
        d = by[s]
        per_c = d.groupby("creator_id").size()
        stats["splits"][s] = {
            "images": len(d), "contributors": int(d["creator_id"].nunique()),
            "sequences": int(d["sequence"].nunique()),
            "images_per_contributor_median_max": [float(per_c.median()), int(per_c.max())],
            "per_country": {c: {"images": int(len(g)), "contributors": int(g["creator_id"].nunique()),
                                "eff_contributors": round(eff_contributors(g["creator_id"]), 1)}
                            for c, g in d.groupby("country")},
        }
    os.makedirs(a.out_dir, exist_ok=True)
    man.to_csv(os.path.join(a.out_dir, "manifest.csv"), index=False)
    with open(os.path.join(a.out_dir, "classes.json"), "w") as f:
        json.dump(classes, f)
    with open(os.path.join(a.out_dir, "subset_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    for s in SPLITS:
        st = stats["splits"][s]
        imgs = [v["images"] for v in st["per_country"].values()]
        effs = [v["eff_contributors"] for v in st["per_country"].values()]
        print(f"{s:12s} {st['images']:7,d} images  {st['contributors']:6,d} contributors  "
              f"per country {min(imgs)}-{max(imgs)} images, eff. contributors {min(effs):.0f}-{max(effs):.0f}")
    print("checks:", checks)


if __name__ == "__main__":
    main()
