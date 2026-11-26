"""Grid of example images from a few visually different countries (read straight from the zip).

Usage:
  python -I src/fig_examples.py --zip data/raw/images_test/04.zip \
      --index data/processed/shard_index.csv --figdir figures \
      --countries JP,BR,RU,US,ZA,FI --per-country 2
"""
import argparse
import io
import os
import sys
import zipfile

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402
from countries import name_of  # noqa: E402

Image.MAX_IMAGE_PIXELS = 50_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--countries", required=True)
    ap.add_argument("--per-country", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    style.apply()
    import matplotlib.pyplot as plt

    df = pd.read_csv(a.index, dtype={"id": str, "country": str}, keep_default_na=False)
    codes = a.countries.split(",")
    R = a.per_country
    C, W, gap = len(codes), 7.2, 0.04
    tile_w = W * 0.98 / (C + (C - 1) * gap)
    rows_h = R * tile_w * 3 / 4 * (1 + gap)
    H = rows_h + 0.5  # 0.5 in for suptitle + column titles
    fig, axes = plt.subplots(R, C, figsize=(W, H), gridspec_kw={"wspace": gap, "hspace": gap})
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.5 / H)
    with zipfile.ZipFile(a.zip) as zf:
        for j, c in enumerate(codes):
            sub = df[df.country == c]
            pick = sub.sample(n=min(R, len(sub)), random_state=a.seed)
            for i in range(R):
                ax = axes[i, j] if R > 1 else axes[j]
                ax.axis("off")
                if i >= len(pick):
                    continue
                im = Image.open(io.BytesIO(zf.read(pick.iloc[i]["member"]))).convert("RGB")
                im.thumbnail((400, 400))
                w, h = im.size
                s = min(w, int(h * 4 / 3))  # centre-crop to 4:3 so tiles line up
                im = im.crop(((w - s) // 2, 0, (w - s) // 2 + s, h))
                ax.imshow(im)
                if i == 0:
                    ax.set_title(name_of(c), fontsize=8.5, loc="center", fontweight="normal")
    fig.suptitle("Example test-split images (OSV-5M shard images/test/04.zip)",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold", y=1 - 0.04 / H, va="top")
    style.save(fig, a.figdir, "fig_c_examples")


if __name__ == "__main__":
    main()
