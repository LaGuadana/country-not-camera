"""Evaluate a checkpoint on the held-out splits and export features for the contributor probe.

For each split: top-1, top-3, balanced accuracy (mean per-class recall) with 95 % CIs from a
contributor-clustered bootstrap, per-class recall and the confusion matrix. The headline number
is the contributor gap: test_shared minus test (unseen trips of seen contributors vs. unseen
contributors).

Writes to --out-dir (default: <checkpoint dir>/eval): metrics.json, predictions.csv,
confusion_<split>.csv and, with --features, features_<split>.npz (512-d pooled features with
ids, labels and contributor ids).

Usage:
  python scripts/evaluate.py --ckpt runs/ft_imagenet/best.pt --features
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocnn.data import AugConfig, Prep, make_loader, open_data  # noqa: E402
from geocnn.engine import predict  # noqa: E402
from geocnn.metrics import ci, cluster_bootstrap, summarize  # noqa: E402
from geocnn.model import CountryNet  # noqa: E402


def fmt_ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=None, help="default: the training run's --data-dir")
    ap.add_argument("--splits", default="test,test_shared,val,val_shared")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--features", action="store_true", help="save pooled features for the contributor probe")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    targs = cfg["args"]
    data_dir = a.data_dir or targs["data_dir"]
    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "eval")
    os.makedirs(out_dir, exist_ok=True)

    man, classes, reader, masks = open_data(data_dir)
    if classes != cfg["classes"]:
        raise SystemExit("class list of --data-dir differs from the checkpoint's")
    aug = AugConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["aug"].items()})
    if aug.fingerprint and aug.static_mask and masks is None:
        print("WARNING: the model was trained with static masks but --data-dir has no masks.npz")
    # architecture only; the weights come from the checkpoint
    model = CountryNet(len(classes), init="scratch", dropout=targs["dropout"])
    model.load_state_dict(ck["model"])
    model.to(device)
    prep = Prep(aug, train=False).to(device)
    channels_last = device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    results = {"checkpoint": os.path.abspath(a.ckpt), "epoch": ck.get("epoch"), "classes": classes,
               "train_args": targs, "splits": {}}
    boot = {}
    preds = []
    for split in a.splits.split(","):
        rows = man[man["split"] == split].reset_index(drop=True)
        if rows.empty:
            print(f"skip {split}: no images")
            continue
        loader = make_loader(rows, reader, False, aug, masks, a.batch_size, a.workers)
        r = predict(model, loader, device, prep, features=a.features, channels_last=channels_last)
        rows = rows.iloc[r["idx"]].reset_index(drop=True)
        m = summarize(r["logits"], r["labels"], len(classes))
        correct = (r["logits"].argmax(1) == r["labels"]).astype(np.float64)
        b_top1, b_bacc = cluster_bootstrap(correct, r["labels"], rows["creator_id"].to_numpy(),
                                           len(classes), n_boot=a.n_boot)
        boot[split] = (b_top1, b_bacc)
        m["top1_ci95"], m["balanced_acc_ci95"] = ci(b_top1), ci(b_bacc)
        m["contributors"] = int(rows["creator_id"].nunique())
        pd.DataFrame(m.pop("confusion"), index=classes, columns=classes).to_csv(
            os.path.join(out_dir, f"confusion_{split}.csv"))
        m["per_class_recall"] = dict(zip(classes, m["per_class_recall"]))
        results["splits"][split] = m
        prob = torch.softmax(torch.from_numpy(r["logits"]), 1).numpy()
        top3 = np.argsort(-prob, axis=1)[:, :3]
        preds.append(pd.DataFrame({
            "id": rows["id"], "split": split, "creator_id": rows["creator_id"], "country": rows["country"],
            "pred": [classes[k] for k in top3[:, 0]], "p_true": prob[np.arange(len(prob)), r["labels"]],
            "top3": ["|".join(classes[k] for k in t) for t in top3]}))
        if a.features:
            np.savez_compressed(os.path.join(out_dir, f"features_{split}.npz"), features=r["features"],
                                ids=rows["id"].to_numpy(str), labels=r["labels"],
                                creator_id=rows["creator_id"].to_numpy(str))
        print(f"{split:12s} n={m['n']:6,d}  top1 {m['top1']:.3f} {fmt_ci(m['top1_ci95'])}  "
              f"bacc {m['balanced_acc']:.3f} {fmt_ci(m['balanced_acc_ci95'])}  top3 {m['top3']:.3f}", flush=True)

    for shared, disjoint in (("test_shared", "test"), ("val_shared", "val")):
        if shared in boot and disjoint in boot:
            s, d = results["splits"][shared], results["splits"][disjoint]
            results[f"gap_{shared}_minus_{disjoint}"] = {
                "top1": s["top1"] - d["top1"], "top1_ci95": ci(boot[shared][0] - boot[disjoint][0]),
                "balanced_acc": s["balanced_acc"] - d["balanced_acc"],
                "balanced_acc_ci95": ci(boot[shared][1] - boot[disjoint][1])}
            g = results[f"gap_{shared}_minus_{disjoint}"]
            print(f"gap {shared} - {disjoint}: top1 {g['top1']:+.3f} {fmt_ci(g['top1_ci95'])}  "
                  f"bacc {g['balanced_acc']:+.3f} {fmt_ci(g['balanced_acc_ci95'])}")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    if preds:
        pd.concat(preds).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
