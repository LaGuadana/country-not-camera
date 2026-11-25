"""EDA on the OSV-5M metadata CSVs (country counts, long tail, coordinate sample).

Reads only the needed columns, in chunks, to keep RAM low.

Usage:
  python -I src/eda_metadata.py --train data/raw/train.csv --test data/raw/test.csv \
      --results results --figdir figures --processed data/processed
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402
from countries import name_of  # noqa: E402

USECOLS = ["latitude", "longitude", "country", "sequence"]
# keep_default_na=False matters: Namibia's ISO code is "NA".
READ_KW = dict(usecols=USECOLS, dtype={"country": str, "sequence": str,
                                        "latitude": "float32", "longitude": "float32"},
               keep_default_na=False, na_values={"latitude": [""], "longitude": [""]})


def scan(path, sample_frac, seed, chunksize=500_000):
    counts = {}
    seqs = set()
    samples = []
    n = 0
    rng = np.random.default_rng(seed)
    for chunk in pd.read_csv(path, chunksize=chunksize, **READ_KW):
        n += len(chunk)
        vc = chunk["country"].value_counts()
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)
        seqs.update(chunk["sequence"].unique().tolist())
        if sample_frac > 0:
            m = rng.random(len(chunk)) < sample_frac
            samples.append(chunk.loc[m, ["latitude", "longitude", "country"]].copy())
        print(f"  {path}: {n:,} rows", flush=True)
    s = pd.Series(counts).sort_values(ascending=False)
    samp = pd.concat(samples) if samples else None
    return n, s, len(seqs), samp


def tail_stats(s):
    tot = int(s.sum())
    return {
        "n_images": tot,
        "n_countries": int((s > 0).sum()),
        "n_countries_ge_100": int((s >= 100).sum()),
        "n_countries_ge_1k": int((s >= 1_000).sum()),
        "n_countries_ge_10k": int((s >= 10_000).sum()),
        "n_countries_ge_100k": int((s >= 100_000).sum()),
        "top1_country": s.index[0], "top1_count": int(s.iloc[0]),
        "top1_share": float(s.iloc[0] / tot),
        "top10_share": float(s.iloc[:10].sum() / tot),
        "top30_share": float(s.iloc[:30].sum() / tot),
        "median_count": float(s.median()),
        "n_countries_lt_100": int((s < 100).sum()),
    }


def fig_counts(s, stats, figdir):
    import matplotlib.pyplot as plt
    top = s.iloc[:30][::-1]
    labels = [f"{name_of(c)} ({c})" for c in top.index]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.55})
    ax.barh(range(len(top)), top.values, color=style.BLUE, height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(left=10 ** np.floor(np.log10(top.min())) )
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("training images (log scale)")
    ax.set_title("Top 30 countries, OSV-5M train split")
    ax.tick_params(axis="y", length=0)

    r = np.arange(1, len(s) + 1)
    ax2.plot(r, s.values, color=style.BLUE, lw=1.4)
    for thr, lab, x, ha, yf, va, sep in [(1_000, "1k", 0.03, "left", 1 / 1.3, "top", "\n"),
                                         (10_000, "10k", 0.98, "right", 1.25, "bottom", " ")]:
        ax2.axhline(thr, color=style.INK2, lw=0.6, ls="--")
        k = int((s >= thr).sum())
        ax2.text(x, thr * yf, f"{k} countries{sep}with ≥{lab}", ha=ha, va=va,
                 transform=ax2.get_yaxis_transform(), color=style.INK2, fontsize=7.5)
    ax2.set_yscale("log")
    ax2.set_xlabel("country rank")
    ax2.set_ylabel("training images (log scale)")
    ax2.set_title(f"All {stats['n_countries']} countries: long tail")
    ax2.text(0.04, 0.04, f"top 10 hold {100 * stats['top10_share']:.1f}%\nof {stats['n_images']:,} images",
             transform=ax2.transAxes, ha="left", va="bottom", fontsize=7.5, color=style.INK2)
    style.save(fig, figdir, "fig_a_country_counts")


def fig_world(samp, figdir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.scatter(samp["longitude"], samp["latitude"], s=0.6, alpha=0.25, color=style.BLUE,
               linewidths=0, rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-60, 81, 30))
    ax.set_xlabel("longitude (°)")
    ax.set_ylabel("latitude (°)")
    ax.set_title(f"Locations of a random sample of {len(samp):,} training images")
    style.save(fig, figdir, "fig_b_world_scatter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--n-sample", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from saved results")
    a = ap.parse_args()
    style.apply()
    if a.plot_only:
        s_tr = pd.read_csv(os.path.join(a.results, "country_counts_train.csv"), keep_default_na=False,
                           dtype={"country": str}).set_index("country")["n_images"]
        with open(os.path.join(a.results, "metadata_stats.json")) as f:
            st = json.load(f)
        fig_counts(s_tr, st["train"], a.figdir)
        fig_world(pd.read_csv(os.path.join(a.processed, "train_coords_sample.csv")), a.figdir)
        return
    os.makedirs(a.results, exist_ok=True)
    os.makedirs(a.processed, exist_ok=True)

    out = {}
    n_te, s_te, nseq_te, _ = scan(a.test, 0, a.seed)
    s_te.rename("n_images").to_csv(os.path.join(a.results, "country_counts_test.csv"), index_label="country")
    out["test"] = {"n_rows": n_te, "n_sequences": nseq_te, **tail_stats(s_te)}

    n_tr, s_tr, nseq_tr, samp = scan(a.train, 1.15 * a.n_sample / 4.9e6, a.seed)
    s_tr.rename("n_images").to_csv(os.path.join(a.results, "country_counts_train.csv"), index_label="country")
    out["train"] = {"n_rows": n_tr, "n_sequences": nseq_tr, **tail_stats(s_tr)}
    out["countries_in_test_not_train"] = sorted(set(s_te.index) - set(s_tr.index))
    out["countries_in_train_not_test"] = sorted(set(s_tr.index) - set(s_te.index))
    out["n_countries_union"] = len(set(s_te.index) | set(s_tr.index))
    out["top10_train"] = [[c, name_of(c), int(v)] for c, v in s_tr.iloc[:10].items()]

    samp = samp.sample(n=min(a.n_sample, len(samp)), random_state=a.seed)
    samp.to_csv(os.path.join(a.processed, "train_coords_sample.csv"), index=False)
    out["coord_sample_n"] = len(samp)

    with open(os.path.join(a.results, "metadata_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    fig_counts(s_tr, out["train"], a.figdir)
    fig_world(samp, a.figdir)


if __name__ == "__main__":
    main()
