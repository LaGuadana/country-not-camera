"""Place or camera? Excess accuracy on new trips of seen photographers, binned by the distance from each
test_shared image to the nearest TRAINING image of the same photographer (haversine on the manifest's GPS).

If the photographer advantage came from recognising familiar places, it would vanish for trips far from
anything the model saw by that photographer; if it comes from the camera, it persists at any distance.
Excess = correct - the same model's accuracy on that country in test (unseen photographers), averaged per
bin; 95 % CIs from a photographer-clustered bootstrap. Only for runs trained on the manifest train split.

  python -I scripts/place_vs_camera.py runs/ft_imagenet runs/ft_imagenet_fp --out results/final/place_vs_camera.json
"""
import argparse
import json

import numpy as np
import pandas as pd

R = 6371.0
BINS = [(0, 25, "<25 km"), (25, 100, "25-100 km"), (100, 500, "100-500 km"), (500, 1e9, ">=500 km")]


def nearest_train_km(man):
    tr = man[man.split == "train"]
    ts = man[man.split == "test_shared"]
    out = pd.Series(np.nan, index=ts.id)
    lat_t = np.radians(tr.latitude.astype(float).to_numpy()); lon_t = np.radians(tr.longitude.astype(float).to_numpy())
    by_c = {c: idx for c, idx in tr.groupby("creator_id").indices.items()}
    for c, g in ts.groupby("creator_id"):
        idx = by_c.get(c)
        if idx is None:
            continue
        la = np.radians(g.latitude.astype(float).to_numpy())[:, None]; lo = np.radians(g.longitude.astype(float).to_numpy())[:, None]
        d = 2 * R * np.arcsin(np.sqrt(np.sin((lat_t[idx] - la) / 2) ** 2
                                      + np.cos(la) * np.cos(lat_t[idx]) * np.sin((lon_t[idx] - lo) / 2) ** 2))
        out.loc[g.id.to_numpy()] = d.min(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    man = pd.read_csv("data/c32/manifest.csv", dtype=str, keep_default_na=False)
    dist = nearest_train_km(man)
    res = {"distance_quantiles_km": {q: float(dist.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}, "runs": {}}
    rng = np.random.default_rng(0)
    for run in a.runs:
        p = pd.read_csv(f"{run}/eval/predictions.csv", dtype=str, keep_default_na=False)
        p["correct"] = (p.pred == p.country).astype(float)
        acc_c = p[p.split == "test"].groupby("country").correct.mean()
        q = p[p.split == "test_shared"].copy()
        q["km"] = q.id.map(dist)
        q["ex"] = q.correct - q.country.map(acc_c)
        rows = []
        for lo, hi, lab in BINS:
            b = q[(q.km >= lo) & (q.km < hi)]
            codes, inv = np.unique(b.creator_id, return_inverse=True)
            w = rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)), size=a.n_boot)[:, inv]
            d = (w @ b.ex.to_numpy()) / w.sum(1)
            rows.append({"bin": lab, "images": int(len(b)), "photographers": int(len(codes)),
                         "excess": float(b.ex.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]})
            print(f"{run:22s} {lab:11s} n={len(b):5d} phot={len(codes):4d} excess {100*b.ex.mean():+5.1f} pp "
                  f"[{100*np.percentile(d,2.5):+.1f}, {100*np.percentile(d,97.5):+.1f}]")
        res["runs"][run] = rows
    print("distance quantiles km:", {k: round(v) for k, v in res["distance_quantiles_km"].items()})
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
