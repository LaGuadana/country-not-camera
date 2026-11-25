"""The proposal's classical baseline: colour histograms + HOG, PCA, k-nearest neighbours.

Hand-crafted features in the spirit of IM2GPS (Hays and Efros 2008): a 16x4x4 HSV colour
histogram (L1-normalised, square-rooted) and HOG (Dalal and Triggs 2005; 9 orientations,
16x16 px cells, 2x2 cell blocks -> 2,268 dims on our 128x170 images), concatenated,
standardised and reduced with PCA fitted on the training sample (the curse of
dimensionality at work; Beyer et al. 1999). Classification is distance-weighted kNN.

The training gallery is the SAME sample as the linear probe (first 300 manifest images per
country, 9,600 images), so the two baselines differ only in the representation: hand-crafted
descriptors versus frozen ImageNet features. k and the distance are chosen on val (unseen
contributors); test and test_shared are then scored once, with contributor-clustered
bootstrap CIs and the shared-disjoint gap, in the same format as the CNN runs.

Usage:
  python -I scripts/classical_baseline.py --data-dir data/c32 --out-dir runs/classical
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from geocnn.images import PackedReader, STORE_H, STORE_W  # noqa: E402
from geocnn.metrics import ci, cluster_bootstrap  # noqa: E402

_H_BINS, _S_BINS, _V_BINS = 16, 4, 4
_R = None  # worker-global reader


def _features_one(args):
    image_id, = args
    from PIL import Image
    from skimage.color import rgb2gray, rgb2hsv
    from skimage.feature import hog
    import io

    im = Image.open(io.BytesIO(_R.read(image_id))).convert("RGB")
    a = np.asarray(im)
    hsv = rgb2hsv(a)
    hist, _ = np.histogramdd(hsv.reshape(-1, 3),
                             bins=(_H_BINS, _S_BINS, _V_BINS),
                             range=((0, 1), (0, 1), (0, 1)))
    h = np.sqrt(hist.ravel() / max(hist.sum(), 1))
    g = hog(rgb2gray(a), orientations=9, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    return np.r_[h, g]


def extract(rows, reader, workers, chunk=256):
    ids = rows["id"].tolist()
    global _R
    _R = reader
    with Pool(workers, initializer=_init, initargs=(reader.bin_path,)) as pool:
        feats = []
        for i in range(0, len(ids), chunk):
            feats.extend(pool.map(_features_one, [(i_,) for i_ in ids[i:i + chunk]], chunksize=16))
            if (i // chunk) % 20 == 0:
                print(f"  {i + len(ids[i:i+chunk]):,}/{len(ids):,}", flush=True)
    return np.asarray(feats, dtype=np.float32)


def _init(bin_path):
    global _R
    _R = PackedReader(bin_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/c32")
    ap.add_argument("--out-dir", default="runs/classical")
    ap.add_argument("--per-class", type=int, default=300, help="training gallery per country")
    ap.add_argument("--pca-components", type=int, default=256)
    ap.add_argument("--ks", default="1,15")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    man = pd.read_csv(os.path.join(a.data_dir, "manifest.csv"),
                      dtype={"id": str, "creator_id": str}, keep_default_na=False)
    reader = PackedReader(os.path.join(a.data_dir, "images.bin"))
    man = man[man["id"].astype(object).isin(reader.row.keys())].reset_index(drop=True)
    train = man[man["split"] == "train"]
    gal = train[train.groupby("label").cumcount() < a.per_class].reset_index(drop=True)

    feats = {}
    for name, rows in (("train", gal), ("val", man[man["split"] == "val"]),
                       ("test", man[man["split"] == "test"]),
                       ("test_shared", man[man["split"] == "test_shared"])):
        path = os.path.join(a.out_dir, f"features_{name}.npz")
        if os.path.exists(path):
            z = np.load(path)
            feats[name] = z["features"]
            print(f"{name}: loaded {len(feats[name]):,} cached features")
        else:
            print(f"extracting HSV+HOG for {name}: {len(rows):,} images", flush=True)
            feats[name] = extract(rows, reader, a.workers)
            np.savez_compressed(path, features=feats[name], ids=rows["id"].to_numpy(str),
                                labels=rows["label"].to_numpy(), creator_id=rows["creator_id"].to_numpy(str))

    sc = StandardScaler().fit(feats["train"])
    pca = PCA(n_components=min(a.pca_components, len(feats["train"]) - 1), random_state=0)
    pca.fit(sc.transform(feats["train"]))
    X = {k: pca.transform(sc.transform(v)) for k, v in feats.items()}
    y = {k: np.load(os.path.join(a.out_dir, f"features_{k}.npz"))["labels"] for k in feats}
    var = float(pca.explained_variance_ratio_.sum())
    print(f"PCA: {pca.n_components_} components, {var:.2%} of gallery variance "
          f"(from {feats['train'].shape[1]} dims)")

    classes = sorted(train["country"].unique())
    label = {c: i for i, c in enumerate(classes)}
    ks = [int(k) for k in a.ks.split(",")]
    best = None
    for k in ks:
        for metric in ("euclidean", "cosine"):
            clf = KNeighborsClassifier(k, weights="distance", metric=metric, n_jobs=a.workers)
            clf.fit(X["train"], y["train"])
            acc = float((clf.predict(X["val"]) == y["val"]).mean())
            print(f"val top-1: k={k} {metric} -> {acc:.4f}")
            if best is None or acc > best[0]:
                best = (acc, k, metric)
    _, k, metric = best
    clf = KNeighborsClassifier(k, weights="distance", metric=metric, n_jobs=a.workers)
    clf.fit(X["train"], y["train"])

    out = {"k": k, "metric": metric, "pca_components": pca.n_components_,
           "pca_variance": var, "gallery_images": int(len(y["train"])), "splits": {}}
    preds = []
    boot = {}
    rows_by = {s: man[man["split"] == s].reset_index(drop=True) for s in ("test", "test_shared")}
    for split in ("test", "test_shared"):
        rows = rows_by[split]
        pred = clf.predict(X[split])
        correct = (pred == y[split]).astype(np.float64)
        b_top1, b_bacc = cluster_bootstrap(correct, y[split], rows["creator_id"].to_numpy(),
                                           len(classes), n_boot=a.n_boot)
        boot[split] = (b_top1, b_bacc)
        top3 = np.argsort(-clf.predict_proba(X[split]), axis=1)[:, :3]
        out["splits"][split] = {
            "n": int(len(rows)), "top1": float(correct.mean()),
            "top1_ci95": ci(b_top1), "top3": float((top3 == y[split][:, None]).any(1).mean()),
            "contributors": int(rows["creator_id"].nunique())}
        preds.append(pd.DataFrame({"id": rows["id"], "split": split,
                                   "creator_id": rows["creator_id"], "country": rows["country"],
                                   "pred": [classes[i] for i in pred]}))
    gap = boot["test_shared"][1] - boot["test"][1]
    out["gap_test_shared_minus_test"] = {"top1": out["splits"]["test_shared"]["top1"] - out["splits"]["test"]["top1"],
                                         "top1_ci95": ci(boot["test_shared"][0] - boot["test"][0]),
                                         "balanced_acc": float(np.mean(gap)), "balanced_acc_ci95": ci(gap)}
    with open(os.path.join(a.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    pd.concat(preds).to_csv(os.path.join(a.out_dir, "predictions.csv"), index=False)
    for split in ("test", "test_shared"):
        d = out["splits"][split]
        print(f"{split:12s} n={d['n']:,}  top1 {d['top1']:.4f} {d['top1_ci95']}  top3 {d['top3']:.4f}")
    g = out["gap_test_shared_minus_test"]
    print(f"gap test_shared - test: top1 {g['top1']:+.4f} {g['top1_ci95']}")
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
