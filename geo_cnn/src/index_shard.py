"""Index one OSV-5M image shard (zip) without extracting it.

For every JPEG in the zip: id, member path, width, height (header only, no decode),
joined with the split metadata (country, sequence, lat/lon, creator, ...).
Also writes the image-resolution figure (fig_d).

Usage:
  python -I src/index_shard.py --zip data/raw/images_test/04.zip --meta data/raw/test.csv \
      --out data/processed/shard_index.csv --figdir figures --results results
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

Image.MAX_IMAGE_PIXELS = 50_000_000  # guard against decompression bombs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    style.apply()

    rows = []
    with zipfile.ZipFile(a.zip) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".jpg", ".jpeg")):
                continue
            iid = os.path.splitext(os.path.basename(info.filename))[0]
            try:
                with zf.open(info) as fh:
                    im = Image.open(fh)
                    w, h = im.size
            except Exception as e:  # corrupt member
                print("skip", info.filename, e)
                continue
            rows.append((iid, info.filename, w, h, info.file_size))
            if len(rows) % 2000 == 0:
                print(f"  indexed {len(rows):,}", flush=True)
    idx = pd.DataFrame(rows, columns=["id", "member", "width", "height", "bytes"])

    cols = ["id", "latitude", "longitude", "country", "sequence", "creator_id",
            "captured_at", "quadtree_10_1000"]
    meta_parts = []
    ids = set(idx["id"])
    for ch in pd.read_csv(a.meta, usecols=cols, dtype={"id": str, "country": str, "sequence": str,
                                                       "creator_id": str},
                          keep_default_na=False, na_values={"latitude": [""], "longitude": [""]},
                          chunksize=200_000):
        meta_parts.append(ch[ch["id"].isin(ids)])
    meta = pd.concat(meta_parts)
    df = idx.merge(meta, on="id", how="left", validate="one_to_one")
    n_unmatched = int(df["country"].isna().sum())
    df = df.dropna(subset=["country"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)

    res = df.groupby(["width", "height"]).size().sort_values(ascending=False)
    stats = {
        "zip": os.path.basename(a.zip),
        "zip_bytes": os.path.getsize(a.zip),
        "n_images_in_zip": len(idx),
        "n_matched_to_metadata": len(df),
        "n_unmatched": n_unmatched,
        "n_countries_in_shard": int(df["country"].nunique()),
        "n_sequences_in_shard": int(df["sequence"].nunique()),
        "n_creators_in_shard": int(df["creator_id"].nunique()),
        "width_min_median_max": [int(df.width.min()), float(df.width.median()), int(df.width.max())],
        "height_min_median_max": [int(df.height.min()), float(df.height.median()), int(df.height.max())],
        "share_landscape": float((df.width > df.height).mean()),
        "mean_jpeg_kb": float(df.bytes.mean() / 1024),
        "top_resolutions": [[f"{w}x{h}", int(n), float(n / len(df))] for (w, h), n in res.iloc[:8].items()],
        "top15_countries_in_shard": df["country"].value_counts().iloc[:15].to_dict(),
    }
    os.makedirs(a.results, exist_ok=True)
    with open(os.path.join(a.results, "shard_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.35})
    top = res.iloc[:8][::-1]
    ax1.barh(range(len(top)), top.values / len(df) * 100, color=style.BLUE, height=0.7)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels([f"{w} × {h}" for (w, h) in top.index])
    ax1.grid(axis="y", visible=False)
    ax1.tick_params(axis="y", length=0)
    ax1.set_xlabel("share of images (%)")
    ax1.set_title("Most common resolutions (w × h, px)")
    ar = (df.width / df.height).clip(0.3, 3.0)
    ax2.hist(ar, bins=np.linspace(0.3, 3.0, 55), color=style.BLUE)
    ax2.set_xlabel("aspect ratio (width / height)")
    ax2.set_ylabel("images")
    ax2.set_title(f"Aspect ratio, n = {len(df):,}")
    style.save(fig, a.figdir, "fig_d_resolution")


if __name__ == "__main__":
    main()
