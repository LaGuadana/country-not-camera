"""How far do the wrong answers land? Great-circle distance of misclassifications.

For every evaluated run, each misclassified test image is matched with the great-circle
distance (haversine) between its GPS position and the centroid of the *predicted* country.
The centroid is the normalised mean of the unit vectors of all of that country's images in
the manifest, so it is a robust interior point, and the distance is a lower bound on the
geographic plausibility of the error. For scale, the same distance is reported for correct
predictions (roughly the country's radius) and for a majority-class predictor.

Usage:
  python -I scripts/error_geography.py runs/probe runs/cpu_ft runs/stage_3
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

EARTH_R = 6371.0


def centroids(manifest):
    """Country -> (lat, lon) centroid from all manifest rows of that country."""
    lat = np.radians(manifest["latitude"].to_numpy(float))
    lon = np.radians(manifest["longitude"].to_numpy(float))
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
    out = {}
    for country, g in manifest.groupby("country"):
        v = xyz[g.index].mean(0)
        v /= np.linalg.norm(v)
        out[country] = (np.degrees(np.arcsin(v[2])),
                        np.degrees(np.arctan2(v[1], v[0])))
    return out


def haversine(lat1, lon1, lat2, lon2):
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(h))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data-dir", default="data/c32")
    ap.add_argument("--out", default="runs/error_geography.json")
    a = ap.parse_args()
    man = pd.read_csv(os.path.join(a.data_dir, "manifest.csv"),
                      dtype={"id": str, "creator_id": str}, keep_default_na=False)
    man = man.reset_index(drop=True)
    cent = centroids(man)
    pos = man.set_index("id")[["latitude", "longitude"]]

    out = {}
    for run in a.runs:
        p = pd.read_csv(os.path.join(run, "eval", "predictions.csv"),
                        dtype=str, keep_default_na=False)
        p = p[p["split"] == "test"].set_index("id")
        ll = pos.loc[p.index]
        dist = pd.Series(0.0, index=p.index)
        for country, g in p.groupby("pred"):
            c = cent[country]
            dist.loc[g.index] = haversine(ll.loc[g.index, "latitude"].to_numpy(),
                                          ll.loc[g.index, "longitude"].to_numpy(), c[0], c[1])
        wrong = p["pred"] != p["country"]
        row = {"n_test": int(len(p)), "error_rate": float(wrong.mean()),
               "median_km_errors": float(dist[wrong].median()),
               "median_km_correct": float(dist[~wrong].median()),
               "share_errors_over_1500km": float((dist[wrong] > 1500).mean()),
               "share_errors_over_4000km": float((dist[wrong] > 4000).mean())}
        out[os.path.basename(run)] = row
        print(f"{os.path.basename(run):10s} errors {row['error_rate']:.3f}: median distance "
              f"{row['median_km_errors']:6.0f} km (correct: {row['median_km_correct']:4.0f} km), "
              f">1500 km {row['share_errors_over_1500km']:.2f}, >4000 km {row['share_errors_over_4000km']:.2f}",
              flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
