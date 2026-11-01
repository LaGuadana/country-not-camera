"""k-nearest-neighbour country classification on the exported pooled features.

A non-parametric alternative model on the same 512-d features as the trained heads, and a
mechanistic probe of the contributor shortcut in retrieval form:

  gallery val        -> query test         all contributors disjoint from each other
                                                and from training (hardest protocol);
  gallery val_shared -> query test_shared  gallery contributors are the query images'
                                                training photographers.

For the shared protocol the nearest neighbour is decomposed by whether it comes from the
same contributor as the query; accuracy conditional on each case shows whether the
retrieval shortcut (matching a photographer's other trips) or generic country matching
drives correct answers. Features are z-scored with gallery statistics and compared by
cosine; class scores are distance-weighted votes over the k nearest gallery images.

Usage:
  python -I scripts/knn_probe.py runs/cpu_ft runs/stage_3
"""
import argparse
import json
import os

import numpy as np
from sklearn.preprocessing import StandardScaler


def knn_predict(gal_X, gal_y, q_X, ks):
    """Cosine kNN with inverse-distance voting; returns {k: predicted labels}."""
    sc = StandardScaler().fit(gal_X)
    G = sc.transform(gal_X)
    G = G / np.linalg.norm(G, axis=1, keepdims=True)
    Q = sc.transform(q_X)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    kmax = max(ks)
    preds, nn = {}, None
    for i in range(0, len(Q), 2048):
        sim = Q[i:i + 2048] @ G.T
        idx = np.argpartition(-sim, kmax - 1, axis=1)[:, :kmax]
        rows = np.arange(len(idx))[:, None]
        d = 1.0 - np.take_along_axis(sim, idx, 1)            # cosine distance
        order = np.argsort(d, 1)
        idx, d = np.take_along_axis(idx, order, 1), np.take_along_axis(d, order, 1)
        nn = idx[:, 0] if nn is None else np.r_[nn, idx[:, 0]]
        w = 1.0 / np.maximum(d, 1e-6)
        for k in ks:
            scores = np.zeros((len(idx), gal_y.max() + 1))
            np.add.at(scores, (np.repeat(np.arange(len(idx)), k), gal_y[idx[:, :k].ravel()]),
                      w[:, :k].ravel())
            preds[k] = np.r_[preds.get(k, np.array([], int)), scores.argmax(1)]
    return preds, nn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ks", default="1,15")
    ap.add_argument("--out", default="runs/knn_probe.json")
    a = ap.parse_args()
    ks = [int(k) for k in a.ks.split(",")]

    out = {"ks": ks, "protocols": {}}
    for run in a.runs:
        base = os.path.join(run, "eval")
        for gallery, query in (("val", "test"), ("val_shared", "test_shared")):
            try:
                G = np.load(os.path.join(base, f"features_{gallery}.npz"))
                Q = np.load(os.path.join(base, f"features_{query}.npz"))
            except FileNotFoundError:
                print(f"{run} {gallery}->{query}: features missing; skipped")
                continue
            preds, nn = knn_predict(G["features"], G["labels"], Q["features"], ks)
            same = G["creator_id"].astype(str)[nn] == Q["creator_id"].astype(str)
            row = {"gallery": gallery, "query": query, "n_query": int(len(Q)),
                   "nn_same_contributor": float(same.mean())}
            for k, p in preds.items():
                row[f"top1_k{k}"] = float((p == Q["labels"]).mean())
                row[f"top1_k{k}_nn_same"] = float((p[same] == Q["labels"][same]).mean()) if same.any() else None
                row[f"top1_k{k}_nn_diff"] = float((p[~same] == Q["labels"][~same]).mean()) if (~same).any() else None
                row[f"n_same"] = int(same.sum())
            key = f"{os.path.basename(run)}:{gallery}->{query}"
            out["protocols"][key] = row
            fmt_same = (f"{row['top1_k1_nn_same']:.3f}" if row["top1_k1_nn_same"] is not None else "-")
            fmt_diff = (f"{row['top1_k1_nn_diff']:.3f}" if row["top1_k1_nn_diff"] is not None else "-")
            print(f"{os.path.basename(run):10s} {gallery:10s}->{query:11s} "
                  f"1-NN {row['top1_k1']:.3f}  15-NN {row['top1_k15']:.3f}  "
                  f"NN same contributor {row['nn_same_contributor']:.3f} "
                  f"(top1 | same {fmt_same} vs diff {fmt_diff})", flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
