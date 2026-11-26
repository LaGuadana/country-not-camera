"""Streaming feature extraction straight from the shard zip (no extraction to disk).

--kind hsv      : joint HSV colour histogram, 16 hue x 4 sat x 4 value = 256 bins,
                  computed on a <=256 px thumbnail, L1-normalised.
--kind resnet18 : frozen torchvision ResNet-18 (ImageNet-1k V1 weights), CPU,
                  resize short side to 224 + centre crop 224, ImageNet normalisation,
                  512-d global-average-pooled penultimate features.

Usage:
  python -I src/features.py --kind hsv --zip data/raw/images_test/04.zip \
      --split data/processed/split.csv --out data/processed/feat_hsv.npz
"""
import argparse
import io
import json
import os
import time
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = 50_000_000


def load_rgb(zf, member, draft=None):
    im = Image.open(io.BytesIO(zf.read(member)))
    if draft:
        im.draft("RGB", draft)
    return im.convert("RGB")


def hsv_hist(im, bins=(16, 4, 4)):
    im = im.copy()
    im.thumbnail((256, 256))
    hsv = np.asarray(im.convert("HSV"), dtype=np.uint16).reshape(-1, 3)
    h = hsv[:, 0] * bins[0] // 256
    s = hsv[:, 1] * bins[1] // 256
    v = hsv[:, 2] * bins[2] // 256
    idx = (h * bins[1] + s) * bins[2] + v
    hist = np.bincount(idx, minlength=bins[0] * bins[1] * bins[2]).astype(np.float32)
    return hist / hist.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["hsv", "resnet18"], required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    df = pd.read_csv(a.split, dtype={"id": str}, keep_default_na=False)
    members = df["member"].tolist()
    t0 = time.time()
    t_fwd = 0.0
    with zipfile.ZipFile(a.zip) as zf:
        if a.kind == "hsv":
            feats = np.stack([hsv_hist(load_rgb(zf, m)) for m in members])
        else:
            import torch
            import torchvision.transforms as T
            from torchvision.models import ResNet18_Weights, resnet18
            torch.set_num_threads(a.threads)
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Identity()
            model.eval()
            t0 = time.time()  # time only extraction, not the one-off weight download / model build
            tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            out = []
            with torch.inference_mode():
                for i in range(0, len(members), a.batch):
                    x = torch.stack([tf(load_rgb(zf, m, draft=(448, 448))) for m in members[i:i + a.batch]])
                    tf0 = time.time()
                    out.append(model(x).numpy().astype(np.float32))
                    t_fwd += time.time() - tf0
                    done = min(i + a.batch, len(members))
                    if (i // a.batch) % 10 == 0:
                        el = time.time() - t0
                        print(f"  {done}/{len(members)}  {done / el:.1f} img/s", flush=True)
            feats = np.concatenate(out)
    el = time.time() - t0
    np.savez_compressed(a.out, ids=df["id"].values.astype(str), X=feats)
    info = {"kind": a.kind, "n": len(members), "dim": int(feats.shape[1]), "seconds": el,
            "img_per_s_end_to_end": len(members) / el, "threads": a.threads}
    if a.kind == "resnet18":
        info["img_per_s_forward_only"] = len(members) / t_fwd
        info["batch"] = a.batch
    with open(os.path.splitext(a.out)[0] + "_timing.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
