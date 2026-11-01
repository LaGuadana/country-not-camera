"""Train the ResNet-18 country classifier: ImageNet fine-tuning or training from scratch.

Model selection uses balanced accuracy on `val` (contributors never seen in training).
`val_shared` (unseen trips of seen contributors) is evaluated alongside; the difference
val_shared - val tracks how much the model leans on contributor identity.

Writes to --out-dir: config.json, metrics.csv (one row per epoch), last.pt, best.pt.
Re-running with --resume continues from last.pt (e.g. after a Kaggle session ends).

Usage (GPU):
  python scripts/train.py --data-dir data/c32 --out-dir runs/ft_imagenet --init imagenet
  python scripts/train.py --data-dir data/c32 --out-dir runs/ft_imagenet_fp --init imagenet --fingerprint
  python scripts/train.py --data-dir data/c32 --out-dir runs/scratch --init scratch
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocnn.data import AugConfig, Prep, make_loader, open_data  # noqa: E402
from geocnn.engine import amp_dtype, predict  # noqa: E402
from geocnn.metrics import summarize  # noqa: E402
from geocnn.model import CountryNet, freeze_backbone, param_groups  # noqa: E402

# Starting points, not tuned: SGD lr for batch 256 (scaled linearly with --batch-size), weight decay.
DEFAULTS = {
    "imagenet": {"lr": 0.02, "weight_decay": 1e-4, "head_lr_mult": 10.0, "warmup_epochs": 1, "epochs": 30},
    "scratch": {"lr": 0.2, "weight_decay": 5e-4, "head_lr_mult": 1.0, "warmup_epochs": 3, "epochs": 60},
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init", choices=["imagenet", "scratch"], default="imagenet")
    ap.add_argument("--weights", default=None,
                    help="local ResNet-18 ImageNet .pth (resnet18-f37072fd.pth); default: torchvision download")
    ap.add_argument("--fingerprint", action="store_true", help="enable the fingerprint-removal pipeline")
    ap.add_argument("--bottom-frac", type=float, default=0.2)
    ap.add_argument("--static-mask", action="store_true",
                    help="with --fingerprint, also blank each contributor's static-pixel mask (masks.npz)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--opt", choices=["sgd", "adamw"], default="sgd")
    ap.add_argument("--lr", type=float, default=None, help="base LR at batch 256 (SGD) or absolute (AdamW)")
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--head-lr-mult", type=float, default=None)
    ap.add_argument("--warmup-epochs", type=float, default=None)
    ap.add_argument("--freeze-epochs", type=int, default=0,
                    help="train only the head for this many epochs first (linear probe, then fine-tune)")
    ap.add_argument("--freeze-stages", type=int, default=0, choices=range(5),
                    help="keep the stem and the first K residual stages frozen throughout (CPU budget runs)")
    ap.add_argument("--init-from", default=None,
                    help="start from this checkpoint's weights, e.g. runs/probe/best.pt (head already fitted)")
    ap.add_argument("--epoch-size", type=int, default=0,
                    help="images per epoch; epochs walk through one random permutation (default: all)")
    ap.add_argument("--split-pass", type=int, default=0,
                    help="set --epoch-size so that this many epochs make one pass over the training set")
    ap.add_argument("--chunk", type=int, default=0,
                    help="train on this chunk of <data-dir>/train_extra.csv (scripts/extend_train.py) "
                         "instead of the manifest's train split")
    ap.add_argument("--replay-floor", type=int, default=1600,
                    help="with --chunk: top up each country to this many images with random earlier "
                         "training images (manifest train split and earlier chunks)")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--class-weighted", action="store_true", help="inverse-frequency class weights in the loss")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--eval-splits", default="val,val_shared")
    ap.add_argument("--select-split", default="val", help="split whose balanced accuracy picks best.pt")
    ap.add_argument("--limit-train", type=int, default=0, help="debug: random subset of training images")
    ap.add_argument("--limit-eval", type=int, default=0, help="debug: random subset of each eval split")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after this many steps")
    ap.add_argument("--time-limit-min", type=float, default=0,
                    help="stop after the epoch that would overrun this budget (resume later with --resume)")
    ap.add_argument("--no-channels-last", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    for k, v in DEFAULTS[a.init].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def lr_factor(step, a, steps_per_epoch):
    """Linear warm-up then cosine decay to 0, counted from the end of the frozen (head-only) phase."""
    if a.freeze_epochs and step < a.freeze_epochs * steps_per_epoch:
        return 1.0
    s = step - a.freeze_epochs * steps_per_epoch
    total = (a.epochs - a.freeze_epochs) * steps_per_epoch
    warm = max(1, int(a.warmup_epochs * steps_per_epoch))
    if s < warm:
        return (s + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))


def chunk_rows(data_dir, man, chunk, floor, seed):
    """Chunk `chunk` of train_extra.csv, plus random earlier training images for every country
    with fewer than `floor` images in the chunk, up to `floor`."""
    extra = pd.read_csv(os.path.join(data_dir, "train_extra.csv"), dtype={"id": str, "creator_id": str},
                        keep_default_na=False)
    new = extra[extra["chunk"] == chunk]
    if new.empty:
        raise SystemExit(f"no chunk {chunk} in {data_dir}/train_extra.csv")
    old = pd.concat([man[man["split"] == "train"], extra[extra["chunk"] < chunk]], ignore_index=True)
    have = new["country"].value_counts()
    parts = [new]
    for country, g in old.groupby("country"):
        need = floor - int(have.get(country, 0))
        if need > 0:
            parts.append(g.sample(min(need, len(g)), random_state=seed * 1_000_003 + chunk))
    rows = pd.concat(parts, ignore_index=True)
    print(f"chunk {chunk}: {len(new):,} new images + {len(rows) - len(new):,} earlier ones "
          f"(countries topped up to {floor})")
    return rows


def evaluate_splits(model, loaders, device, prep, channels_last, num_classes):
    out = {}
    for name, loader in loaders.items():
        r = predict(model, loader, device, prep, channels_last=channels_last)
        m = summarize(r["logits"], r["labels"], num_classes)
        out[name] = {k: m[k] for k in ("n", "top1", "top3", "balanced_acc")}
    return out


def main(argv=None):
    a = parse_args(argv)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    channels_last = device.type == "cuda" and not a.no_channels_last
    os.makedirs(a.out_dir, exist_ok=True)

    man, classes, reader, masks = open_data(a.data_dir)
    aug = AugConfig(fingerprint=a.fingerprint, bottom_frac=a.bottom_frac, static_mask=a.static_mask)
    if a.fingerprint and aug.static_mask and masks is None:
        print("note: no masks.npz in --data-dir; fingerprint removal uses the bottom band only")
    train_rows = chunk_rows(a.data_dir, man, a.chunk, a.replay_floor, a.seed) if a.chunk else man[man["split"] == "train"]
    missing = ~train_rows["id"].astype(object).isin(reader.row.keys())
    if missing.any():
        raise SystemExit(f"{missing.sum():,} training images are not in images.bin yet; run fetch_images.py")
    if a.limit_train:
        train_rows = train_rows.sample(min(a.limit_train, len(train_rows)), random_state=a.seed)
    if a.split_pass:
        a.epoch_size = math.ceil(len(train_rows) / a.split_pass / a.batch_size) * a.batch_size
    eval_rows = {}
    for s in a.eval_splits.split(","):
        rows = man[man["split"] == s]
        if a.limit_eval:
            rows = rows.sample(min(a.limit_eval, len(rows)), random_state=a.seed)
        eval_rows[s] = rows
    if a.select_split not in eval_rows:
        raise SystemExit(f"--select-split {a.select_split} is not among --eval-splits")
    num_classes = len(classes)
    counts = np.bincount(train_rows["label"], minlength=num_classes)
    print(f"device {device}; train {len(train_rows):,} images, {num_classes} classes "
          f"({counts.min()}-{counts.max()} per class); eval " +
          ", ".join(f"{s} {len(r):,}" for s, r in eval_rows.items()), flush=True)

    train_loader = make_loader(train_rows, reader, True, aug, masks, a.batch_size, a.workers, seed=a.seed,
                               epoch_size=a.epoch_size)
    eval_loaders = {s: make_loader(r, reader, False, aug, masks, 2 * a.batch_size, a.workers)
                    for s, r in eval_rows.items()}

    model = CountryNet(num_classes, a.init, a.weights, a.dropout)
    if a.init_from:
        src = torch.load(a.init_from, map_location="cpu", weights_only=False)
        if src["config"]["classes"] != classes:
            raise SystemExit("--init-from checkpoint has a different class list")
        model.load_state_dict(src["model"])
        print(f"initialised from {a.init_from}")
    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    prep_train, prep_eval = Prep(aug, train=True).to(device), Prep(aug, train=False).to(device)
    groups = param_groups(model, a.weight_decay, a.head_lr_mult)
    if a.opt == "sgd":
        base_lr = a.lr * a.batch_size / 256
        opt = torch.optim.SGD(groups, lr=base_lr, momentum=0.9, nesterov=True)
    else:
        base_lr = a.lr
        opt = torch.optim.AdamW(groups, lr=base_lr)
    weight = None
    if a.class_weighted:
        weight = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=a.label_smoothing)
    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)

    steps_per_epoch = len(train_loader) if not a.max_steps else min(a.max_steps, len(train_loader))
    start_epoch, step, best = 0, 0, -1.0
    last_path, best_path = os.path.join(a.out_dir, "last.pt"), os.path.join(a.out_dir, "best.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)  # RNG states must stay on CPU
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
        torch.set_rng_state(ck["rng"]["torch"])
        if device.type == "cuda" and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        random.setstate(ck["rng"]["python"])
        print(f"resumed from epoch {ck['epoch']} (best {a.select_split} balanced acc {best:.4f})")
    config = {"args": vars(a), "aug": asdict(aug), "classes": classes, "base_lr": base_lr,
              "train_images": len(train_rows), "steps_per_epoch": steps_per_epoch,
              "torch": torch.__version__, "device": str(device)}
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    metrics_path = os.path.join(a.out_dir, "metrics.csv")
    t_start = time.time()
    for epoch in range(start_epoch, a.epochs):
        t0 = time.time()
        train_loader.generator.manual_seed(a.seed * 1_000_003 + epoch)  # resumable data order
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        freeze_backbone(model, 4 if epoch < a.freeze_epochs else a.freeze_stages)
        loss_sum = torch.zeros((), device=device)  # accumulated on device: no sync every step
        correct = torch.zeros((), device=device, dtype=torch.long)
        seen = 0
        for i, (x, y, _) in enumerate(train_loader):
            if a.max_steps and i >= a.max_steps:
                break
            f = lr_factor(step, a, steps_per_epoch)
            for g in opt.param_groups:
                g["lr"] = base_lr * f * g["lr_mult"]
            x = prep_train(x.to(device, non_blocking=True))
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(x)
                loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            loss_sum += loss.detach() * len(y)
            correct += (logits.argmax(1) == y).sum()
            seen += len(y)
            if a.log_every and (i + 1) % a.log_every == 0:
                print(f"  epoch {epoch} step {i + 1}/{steps_per_epoch}  loss {loss_sum.item() / seen:.4f}  "
                      f"acc {correct.item() / seen:.4f}  lr {base_lr * f:.5f}  {seen / (time.time() - t0):.0f} img/s",
                      flush=True)
        t_train = time.time() - t0
        loss_sum, correct = loss_sum.item(), correct.item()
        ev = evaluate_splits(model, eval_loaders, device, prep_eval, channels_last, num_classes)
        row = {"epoch": epoch, "lr": base_lr * lr_factor(step - 1, a, steps_per_epoch),
               "frozen_backbone": int(epoch < a.freeze_epochs),
               "train_loss": loss_sum / max(seen, 1), "train_top1": correct / max(seen, 1),
               "train_img_per_s": seen / t_train, "epoch_min": (time.time() - t0) / 60}
        for s, m in ev.items():
            for k in ("top1", "top3", "balanced_acc"):
                row[f"{s}_{k}"] = m[k]
        if "val" in ev and "val_shared" in ev:
            row["shared_minus_disjoint_bacc"] = ev["val_shared"]["balanced_acc"] - ev["val"]["balanced_acc"]
        score = ev[a.select_split]["balanced_acc"]
        is_best = score > best
        best = max(best, score)
        new = not os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
              "epoch": epoch, "step": step, "best": best, "config": config,
              "rng": {"torch": torch.get_rng_state(), "python": random.getstate(),
                      "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}}
        if is_best:  # before last.pt, which already records this epoch as the best
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)
        torch.save(ck, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)
        print(f"epoch {epoch}: train loss {row['train_loss']:.4f} top1 {row['train_top1']:.4f} | " +
              " | ".join(f"{s} top1 {m['top1']:.4f} bacc {m['balanced_acc']:.4f}" for s, m in ev.items()) +
              f" | {row['epoch_min']:.1f} min{'  *best*' if is_best else ''}", flush=True)
        elapsed = (time.time() - t_start) / 60
        if a.time_limit_min and elapsed + row["epoch_min"] > a.time_limit_min and epoch + 1 < a.epochs:
            print(f"stopping after epoch {epoch}: time budget {a.time_limit_min} min; continue with --resume")
            break
    print(f"done; best {a.select_split} balanced acc {best:.4f}; checkpoints in {a.out_dir}")


if __name__ == "__main__":
    main()
