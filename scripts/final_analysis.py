"""Final-report analyses on evaluated runs. Writes results/final/*.json and figures.

  paired   paired contributor-bootstrap changes in test BA and in the gap (test_shared - test)
           between two evaluations of the same images (e.g. fingerprint vs standard, blank vs normal)
  trips    trip-level accuracy: mean log-softmax over the frames of a Mapillary sequence
  gradcam  share of the layer-4 Grad-CAM evidence in the bottom 20 % band, the top 25 % (sky) and
           the centre, on a fixed sample of test / test_shared images; plus an example panel

Usage:
  python -I scripts/final_analysis.py paired runs/ft_imagenet/eval runs/ft_imagenet_fp/eval --name fp_imagenet
  python -I scripts/final_analysis.py trips runs/stage_3/eval_repro runs/ft_imagenet/eval ...
  python -I scripts/final_analysis.py gradcam runs/stage_3/best.pt runs/ft_imagenet/best.pt ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocnn.metrics import ci, cluster_bootstrap  # noqa: E402

OUT = "results/final"
CLASSES = json.load(open("data/c32/classes.json"))
LAB = {c: i for i, c in enumerate(CLASSES)}


def bacc_draws(pred, true, creators, n_boot, seed=1):
    corr = (pred == true).astype(float)
    return cluster_bootstrap(corr, np.array([LAB[c] for c in true]), creators, 32, n_boot, seed)[1]


def paired(a):
    A = pd.read_csv(f"{a.evals[0]}/predictions.csv", dtype=str, keep_default_na=False).set_index(["id", "split"])
    res = {"baseline": a.evals[0], "runs": {}}
    for ev in a.evals[1:]:
        B = pd.read_csv(f"{ev}/predictions.csv", dtype=str, keep_default_na=False).set_index(["id", "split"]).loc[A.index]
        d = {}
        for name, P in (("a", A), ("b", B)):
            d[name] = {}
            for s in ("test", "test_shared"):
                m = A.index.get_level_values(1) == s
                d[name][s] = bacc_draws(P["pred"][m].to_numpy(), A["country"][m].to_numpy(),
                                        A["creator_id"][m].to_numpy(), a.n_boot)
        dt = d["b"]["test"] - d["a"]["test"]
        dsh = d["b"]["test_shared"] - d["a"]["test_shared"]
        dg = dsh - dt
        res["runs"][ev] = {"test_change": float(dt.mean()), "test_change_ci95": ci(dt),
                           "test_shared_change": float(dsh.mean()), "test_shared_change_ci95": ci(dsh),
                           "gap_change": float(dg.mean()), "gap_change_ci95": ci(dg)}
        r = res["runs"][ev]
        print(f"{ev} vs {a.evals[0]}: test {100 * r['test_change']:+.1f} {[round(100 * x, 1) for x in r['test_change_ci95']]}"
              f"  gap {100 * r['gap_change']:+.1f} {[round(100 * x, 1) for x in r['gap_change_ci95']]}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/paired_{a.name}.json", "w"), indent=1)


def trips(a):
    man = pd.read_csv("data/c32/manifest.csv", dtype=str, keep_default_na=False).set_index("id")
    out = {}
    for ev in a.evals:
        out[ev] = {}
        for s in ("test", "test_shared"):
            z = np.load(f"{ev}/logits_{s}.npz")
            L = z["logits"] - z["logits"].max(1, keepdims=True)
            logp = L - np.log(np.exp(L).sum(1, keepdims=True))
            ids = z["ids"].astype(str)
            meta = man.loc[ids]
            df = pd.DataFrame({"seq": meta["sequence"].to_numpy(), "cre": meta["creator_id"].to_numpy(),
                               "country": meta["country"].to_numpy()})
            codes, inv = np.unique(df["seq"], return_inverse=True)
            n = np.bincount(inv)
            S = np.zeros((len(codes), 32))
            np.add.at(S, inv, logp)
            trip_pred = np.array(CLASSES)[S.argmax(1)]
            first = pd.Series(np.arange(len(df))).groupby(inv).first().to_numpy()
            tdf = df.iloc[first].reset_index(drop=True).assign(pred=trip_pred, n=n)
            img_pred = np.array(CLASSES)[logp.argmax(1)]
            multi = n[inv] >= 2  # images in trips with >= 2 frames
            r = {"images": int(len(df)), "trips": int(len(codes)),
                 "frames_per_trip_mean": float(n.mean()), "share_images_in_multiframe_trips": float(multi.mean()),
                 "image_ba": float(np.mean([(img_pred[df.country == c] == c).mean() for c in CLASSES]))}
            dt = bacc_draws(tdf["pred"].to_numpy(), tdf["country"].to_numpy(), tdf["cre"].to_numpy(), a.n_boot)
            r["trip_ba"] = float(np.mean([(tdf.pred[tdf.country == c] == c).mean() for c in CLASSES]))
            r["trip_ba_ci95"] = ci(dt)
            tm = tdf[tdf.n >= 2]
            r["multiframe_trips"] = int(len(tm))
            r["multiframe_trip_ba"] = float(np.nanmean([(tm.pred[tm.country == c] == c).mean()
                                                        for c in CLASSES if (tm.country == c).any()]))
            r["multiframe_single_image_ba"] = float(np.nanmean(
                [(img_pred[multi & (df.country == c).to_numpy()] == c).mean() for c in CLASSES
                 if (multi & (df.country == c).to_numpy()).any()]))
            out[ev][s] = r
            print(f"{ev} {s}: image BA {100 * r['image_ba']:.1f}  trip BA {100 * r['trip_ba']:.1f} "
                  f"{[round(100 * x, 1) for x in r['trip_ba_ci95']]} ({r['trips']} trips, "
                  f"{r['multiframe_trips']} with >=2 frames: {100 * r['multiframe_single_image_ba']:.1f} -> "
                  f"{100 * r['multiframe_trip_ba']:.1f})")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/trips.json", "w"), indent=1)


def gradcam(a):
    import torch
    import torch.nn.functional as F
    from geocnn.data import AugConfig, Prep, make_loader, open_data
    from geocnn.engine import get_device
    from geocnn.explain import grad_cam
    from geocnn.model import CountryNet
    device = get_device()
    man, classes, reader, masks = open_data("data/c32")
    rows = man[man["split"].isin(["test", "test_shared"])].sample(frac=1, random_state=0)
    rows = rows[rows.groupby(["split", "label"]).cumcount() < a.per_class].reset_index(drop=True)
    H, Wd = 128, 170
    yy = np.arange(H)[:, None].repeat(Wd, 1)
    xx = np.arange(Wd)[None, :].repeat(H, 0)
    regions = {"bottom20": yy >= int(0.8 * H), "top25": yy < int(0.25 * H),
               "centre": (yy >= H // 4) & (yy < 3 * H // 4) & (xx >= Wd // 4) & (xx < 3 * Wd // 4)}
    area = {k: float(v.mean()) for k, v in regions.items()}
    summary, maps_keep = {"region_area_share": area, "n_images": int(len(rows)), "models": {}}, {}
    for ck in a.evals:
        c = torch.load(ck, map_location="cpu", weights_only=False)
        aug = AugConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in c["config"]["aug"].items()})
        model = CountryNet(32, init="scratch", dropout=c["config"]["args"].get("dropout", 0.2))
        model.load_state_dict(c["model"])
        model.to(device).eval()
        prep = Prep(aug, train=False).to(device)
        loader = make_loader(rows, reader, False, aug, masks, 64, 2)
        shares, correct, maps = [], [], []
        for x, y, idx in loader:
            cam, logits = grad_cam(model, prep(x.to(device)))
            up = F.interpolate(cam[:, None].float(), size=(H, Wd), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            tot = up.sum((1, 2)) + 1e-8
            shares.append(np.stack([up[:, m].sum(1) / tot for m in regions.values()], 1))
            correct.append((logits.argmax(1).cpu().numpy() == y.numpy()))
            maps.append(up[:8] if len(maps) == 0 else up[:0])
        sh = np.concatenate(shares)
        co = np.concatenate(correct)
        name = os.path.basename(os.path.dirname(ck))
        r = {}
        for i, k in enumerate(regions):
            v = sh[:, i]
            r[k] = {"mean": float(v.mean()), "se": float(v.std() / np.sqrt(len(v))),
                    "correct": float(v[co].mean()), "incorrect": float(v[~co].mean())}
        for s in ("test", "test_shared"):
            m = (rows["split"] == s).to_numpy()
            r[f"bottom20_{s}"] = float(sh[m, 0].mean())
        r["accuracy_on_sample"] = float(co.mean())
        summary["models"][name] = r
        maps_keep[name] = (maps[0], aug)
        print(f"{name}: bottom20 {100 * r['bottom20']['mean']:.1f}% (area 20%)  top25 {100 * r['top25']['mean']:.1f}%  "
              f"centre {100 * r['centre']['mean']:.1f}%  | correct {100 * r['bottom20']['correct']:.1f} / "
              f"incorrect {100 * r['bottom20']['incorrect']:.1f}")
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    json.dump(summary, open(f"{OUT}/gradcam.json", "w"), indent=1)
    # panel: first 6 sample images (loader order = rows order), one row per model
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = 6
    names = list(maps_keep)
    fig, axes = plt.subplots(len(names), k, figsize=(1.45 * k, 1.15 * len(names) + 0.2))
    axes = np.atleast_2d(axes)
    for j in range(k):
        img = np.array(reader.image(rows["id"].iloc[j]))
        for i, n in enumerate(names):
            ax = axes[i, j]
            ax.imshow(img)
            ax.imshow(maps_keep[n][0][j], cmap="jet", alpha=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(n, fontsize=6.5)
            if i == 0:
                ax.set_title(rows["country"].iloc[j], fontsize=7)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/figures/gradcam_panel.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", f"{OUT}/gradcam.json and figures/gradcam_panel")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["paired", "trips", "gradcam"])
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--name", default="cmp")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--per-class", type=int, default=25)
    a = ap.parse_args()
    {"paired": paired, "trips": trips, "gradcam": gradcam}[a.what](a)
