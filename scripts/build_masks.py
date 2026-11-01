"""Estimate a static-pixel mask for every contributor in the manifest (all splits).

A mask is kept only for contributors with enough images from enough different sequences (see
geocnn/fingerprint.static_mask). Masks use no labels, so test contributors get theirs from their
own test images, exactly like training contributors.

Writes <data-dir>/masks.npz (creators, bits, shape) and <data-dir>/mask_stats.csv, and prints how
many contributors and images end up with a non-empty mask.

Usage:
  python -I scripts/build_masks.py --data-dir data/c32
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocnn.data import open_data  # noqa: E402
from geocnn.fingerprint import pack_masks, static_mask  # noqa: E402
from geocnn.images import STORE_H, STORE_W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--std-thresh", type=float, default=10.0)
    ap.add_argument("--min-images", type=int, default=8)
    ap.add_argument("--min-sequences", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=60, help="images per contributor used for the estimate")
    ap.add_argument("--max-area", type=float, default=0.1,
                    help="drop masks covering more of the frame (they catch scene content, not the camera)")
    a = ap.parse_args()

    man, _, reader, _ = open_data(a.data_dir)
    stats, creators, masks = [], [], []
    groups = man.groupby("creator_id", sort=False)
    for k, (creator, g) in enumerate(groups):
        g = g.sample(min(len(g), a.max_images), random_state=0)
        info = {"n_images": len(g), "n_sequences": int(g["sequence"].nunique()), "status": "too_few"}
        if len(g) >= a.min_images and info["n_sequences"] >= a.min_sequences:
            stack = np.stack([np.asarray(reader.image(i)) for i in g["id"]])
            m, info = static_mask(stack, g["sequence"].to_numpy(), std_thresh=a.std_thresh,
                                  min_images=a.min_images, min_sequences=a.min_sequences, max_area=a.max_area)
            if m is not None and m.any():
                creators.append(creator)
                masks.append(m)
        stats.append({"creator_id": creator, "split": g["split"].iloc[0], **info})
        if (k + 1) % 2000 == 0:
            print(f"  {k + 1:,}/{groups.ngroups:,} contributors", flush=True)

    st = pd.DataFrame(stats)
    st.to_csv(os.path.join(a.data_dir, "mask_stats.csv"), index=False)
    np.savez_compressed(os.path.join(a.data_dir, "masks.npz"),
                        creators=np.array(creators, dtype=str),
                        bits=pack_masks(masks) if masks else np.zeros((0, (STORE_H * STORE_W + 7) // 8), np.uint8),
                        shape=np.array([STORE_H, STORE_W]))
    imgs = man.groupby("creator_id").size()
    with_mask = imgs[imgs.index.isin(set(creators))].sum()
    print(st["status"].value_counts().to_string())
    print(f"non-empty masks: {len(creators):,} of {len(st):,} contributors, covering {with_mask:,} of "
          f"{len(man):,} images; mean mask area {np.mean([m.mean() for m in masks]) if masks else 0:.4f}")


if __name__ == "__main__":
    main()
