"""Linear probe: frozen ImageNet ResNet-18 features + multinomial logistic regression.

The pre-trained-CNN reference point at the same 128 x 170 input as the fine-tuned models, cheap
enough for a CPU. Features are extracted once (no augmentation), standardised, and an L2-penalised
softmax regression is fitted with L-BFGS; the penalty is picked on `val` (unseen contributors).
The fitted head is folded back into a CountryNet checkpoint (ImageNet backbone + linear head), so
scripts/evaluate.py and scripts/export_game.py read it like any training run.

Usage:
  python -I scripts/linear_probe.py --data-dir data/c32 --out-dir runs/probe \
      --weights ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocnn.data import AugConfig, Prep, make_loader, open_data  # noqa: E402
from geocnn.engine import predict  # noqa: E402
from geocnn.metrics import summarize  # noqa: E402
from geocnn.model import CountryNet  # noqa: E402


def fit(X, y, num_classes, l2, iters=300):
    W = torch.zeros(num_classes, X.shape[1], requires_grad=True)
    b = torch.zeros(num_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W.T + b, y) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weights", default=None, help="local resnet18-f37072fd.pth (default: download)")
    ap.add_argument("--per-class", type=int, default=300, help="training images per country")
    ap.add_argument("--val-per-class", type=int, default=50)
    ap.add_argument("--l2", default="1e-4,3e-4,1e-3,3e-3,1e-2")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    man, classes, reader, masks = open_data(a.data_dir)
    k = len(classes)
    aug = AugConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CountryNet(k, "imagenet", a.weights).to(device)
    prep = Prep(aug, train=False).to(device)

    def features(split, per_class):
        rows = man[man["split"] == split].sample(frac=1, random_state=a.seed)
        rows = rows[rows.groupby("label").cumcount() < per_class].reset_index(drop=True)
        t0 = time.time()
        r = predict(model, make_loader(rows, reader, False, aug, masks, a.batch_size, a.workers),
                    device, prep, features=True)
        print(f"{split}: {len(rows):,} images, {len(rows) / (time.time() - t0):.0f} img/s", flush=True)
        return torch.from_numpy(r["features"]), torch.from_numpy(r["labels"])

    Xtr, ytr = features("train", a.per_class)
    Xva, yva = features("val", a.val_per_class)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zva = (Xtr - mu) / sd, (Xva - mu) / sd
    results, best = {}, None
    for l2 in (float(v) for v in a.l2.split(",")):
        W, b = fit(Ztr, ytr, k, l2)
        m = summarize((Zva @ W.T + b).numpy(), yva.numpy(), k)
        train_top1 = float(((Ztr @ W.T + b).argmax(1) == ytr).float().mean())
        results[l2] = {"val_top1": m["top1"], "val_balanced_acc": m["balanced_acc"], "val_top3": m["top3"],
                       "train_top1": train_top1}
        print(f"l2 {l2:g}: train top1 {train_top1:.3f}  val top1 {m['top1']:.3f}  "
              f"bacc {m['balanced_acc']:.3f}  top3 {m['top3']:.3f}", flush=True)
        if best is None or m["balanced_acc"] > results[best[0]]["val_balanced_acc"]:
            best = (l2, W, b)

    l2, W, b = best
    with torch.no_grad():  # fold the standardisation into the head: W z + b = (W / sd) x + (b - (W / sd) mu)
        model.head.weight.copy_(W / sd)
        model.head.bias.copy_(b - (W / sd) @ mu)
    config = {"args": {"data_dir": os.path.abspath(a.data_dir), "init": "imagenet", "dropout": 0.2,
                       "probe": True, "per_class": a.per_class, "l2": l2, "seed": a.seed},
              "aug": asdict(aug), "classes": classes}
    torch.save({"model": model.cpu().state_dict(), "epoch": 0, "config": config},
               os.path.join(a.out_dir, "best.pt"))
    with open(os.path.join(a.out_dir, "probe.json"), "w") as f:
        json.dump({"chosen_l2": l2, "by_l2": results, "train_images": len(ytr), "val_images": len(yva)}, f, indent=1)
    print(f"chose l2 {l2:g}; wrote {a.out_dir}/best.pt")


if __name__ == "__main__":
    main()
