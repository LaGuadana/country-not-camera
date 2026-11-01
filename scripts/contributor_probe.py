"""How well do the pooled features encode the *contributor*, not the country?

For every evaluated run with exported features (scripts/evaluate.py --features), sample
pairs of images inside an evaluation split and ask how well the cosine similarity of their
512-d pooled features separates *same-contributor* pairs from *different-contributor*
pairs, measured by ROC-AUC (chance 0.5). Negatives always share the true country, so a
high AUC cannot come from country information. Two variants of the positive pairs:

  any            same contributor, any two images (may share the trip / sequence);
  diff sequence  same contributor but from different Mapillary sequences, so the two
                 photos cannot be near-duplicates of one road; this is the honest test
                 of *photographer* identity (camera, mount, colour profile, routes).

AUC needs no fitting, so it works with the few images per contributor that the budgets
allow. Also reported: for a sample of images, the fraction whose nearest neighbour (cosine,
among the rest of the split) shares the contributor, with the random-pair chance sum_k p_k^2.

  test         contributors the model never saw in training. Signal here means the
               features cluster *any* photographer's images (generic camera style).
  test_shared  contributors seen in training. Higher AUC here than on test is direct
               evidence that fine-tuning memorised the *specific* training photographers.

Usage:
  python -I scripts/contributor_probe.py runs/probe runs/cpu_ft runs/stage_1 \
      runs/stage_2 runs/stage_3
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

CANDIDATE_PAIRS = 4_000_000  # one shot; positives appear at ~sum_k p_k^2 ~ 1 %


def cos(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def draw_pairs(rng, cid_idx, country, seq, n_pairs, kind):
    """`kind`: pos_any | pos_diff_seq | neg_same_country. Returns index arrays (i, j)."""
    n = len(cid_idx)
    i = rng.integers(0, n, size=CANDIDATE_PAIRS)
    j = rng.integers(0, n, size=CANDIDATE_PAIRS)
    if kind == "pos_any":
        ok = (cid_idx[i] == cid_idx[j]) & (i != j)
    elif kind == "pos_diff_seq":
        ok = (cid_idx[i] == cid_idx[j]) & (seq[i] != seq[j])
    elif kind == "neg_same_country":
        ok = (cid_idx[i] != cid_idx[j]) & (country[i] == country[j])
    else:
        raise ValueError(kind)
    ok = np.flatnonzero(ok)[:n_pairs]
    return i[ok], j[ok]


def probe_split(feat_path, manifest, n_pairs, seed):
    z = np.load(feat_path)
    meta = manifest.set_index("id").loc[z["ids"].astype(str)]
    seq_idx = pd.factorize(meta["sequence"].to_numpy())[0]
    codes, cid_idx = np.unique(z["creator_id"].astype(str), return_inverse=True)
    country = z["labels"]
    Xn = cos(z["features"].astype(np.float64))
    rng = np.random.default_rng(seed)
    neg_i, neg_j = draw_pairs(rng, cid_idx, country, seq_idx, n_pairs, "neg_same_country")
    neg = (Xn[neg_i] * Xn[neg_j]).sum(1)
    res = {"neg_pairs": int(len(neg))}
    for variant, kind in (("auc_any", "pos_any"), ("auc_diff_seq", "pos_diff_seq")):
        pi, pj = draw_pairs(rng, cid_idx, country, seq_idx, n_pairs, kind)
        pos = (Xn[pi] * Xn[pj]).sum(1)
        yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        res[variant] = float(roc_auc_score(yy, np.r_[pos, neg]))
        res[f"{variant}_pairs"] = int(len(pos))
    # nearest neighbour (leave-self-out) same-contributor rate on a sample
    m = np.sort(rng.choice(len(Xn), size=min(4000, len(Xn)), replace=False))
    nn = NearestNeighbors(n_neighbors=2).fit(Xn[m])
    _, ind = nn.kneighbors(Xn[m])
    same = float((cid_idx[m][ind[:, 1]] == cid_idx[m]).mean())
    diff_seq = float((seq_idx[m][ind[:, 1]] != seq_idx[m]).mean())
    p = np.bincount(cid_idx, minlength=len(codes)) / len(cid_idx)
    res.update({"n_images": int(len(Xn)), "contributors": int(len(codes)),
                "nn_same_contributor": same, "nn_same_sequence": 1.0 - diff_seq,
                "nn_same_chance": float((p ** 2).sum())})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs with eval/features_<split>.npz")
    ap.add_argument("--data-dir", default="data/c32")
    ap.add_argument("--splits", default="test,test_shared")
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    manifest = pd.read_csv(os.path.join(a.data_dir, "manifest.csv"),
                           dtype={"id": str, "creator_id": str, "sequence": str},
                           keep_default_na=False)

    for run in a.runs:
        out_path = os.path.join(run, "eval", "contributor_probe.json")
        res = {"pairs": a.pairs, "seed": a.seed, "splits": {}}
        for split in a.splits.split(","):
            f = os.path.join(run, "eval", f"features_{split}.npz")
            if not os.path.exists(f):
                print(f"{run} {split}: no features (run evaluate.py --features); skipped")
                continue
            r = probe_split(f, manifest, a.pairs, a.seed)
            res["splits"][split] = r
            print(f"{os.path.basename(run):10s} {split:12s} AUC(any) {r['auc_any']:.3f}  "
                  f"AUC(diff-seq) {r['auc_diff_seq']:.3f}  NN-same {r['nn_same_contributor']:.3f} "
                  f"(chance {r['nn_same_chance']:.3f})", flush=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
