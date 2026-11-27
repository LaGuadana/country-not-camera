"""Which parts of the image drive each country's prediction, and how confusable neighbours differ.

1. Per-country Grad-CAM region shares (end-to-end model, true-class maps) on a fixed sample of test
   images (50 per country): share of evidence in the top quarter (sky, tree line, buildings), the
   central band and the bottom fifth (road surface, markings, bonnet).
2. Panel for two confusable pairs (Austria vs Switzerland, Canada vs United States): the same test image
   with the evidence maps for both candidate countries.
3. Confusion-matrix figure of the end-to-end model on test, countries grouped by region.
Writes results/final/country_features.json and report/final/figures/fig_country_cam.*, fig_pairs.*,
fig_confusion.*.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocnn.data import AugConfig, Prep, make_loader, open_data  # noqa: E402
from geocnn.engine import get_device  # noqa: E402
from geocnn.explain import grad_cam  # noqa: E402
from geocnn.model import CountryNet  # noqa: E402

OUT = "report/final/figures"
REGION_ORDER = ["AT", "BE", "CH", "CZ", "DE", "DK", "ES", "FR", "GB", "GR", "HR", "HU", "IT", "NL", "PL", "PT",
                "SE", "RU", "TR", "BR", "CA", "CO", "MX", "US", "ID", "IN", "JP", "MY", "TH", "TW", "AU", "ET"]
mpl_rc = {"font.size": 7.4, "axes.titlesize": 8.0, "axes.spines.top": False, "axes.spines.right": False,
          "figure.dpi": 200, "savefig.bbox": "tight"}


def load_model(ck, device):
    c = torch.load(ck, map_location="cpu", weights_only=False)
    aug = AugConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in c["config"]["aug"].items()})
    m = CountryNet(32, init="scratch", dropout=c["config"]["args"].get("dropout", 0.2))
    m.load_state_dict(c["model"]); m.to(device).eval()
    return m, aug


def main():
    plt.rcParams.update(mpl_rc)
    device = get_device()
    man, classes, reader, masks = open_data("data/c32")
    lab = {c: i for i, c in enumerate(classes)}
    model, aug = load_model("runs/ft_imagenet/best.pt", device)
    prep = Prep(aug, train=False).to(device)
    H, W = 128, 170
    yy = np.arange(H)[:, None].repeat(W, 1)
    regions = {"top25": yy < H // 4, "middle": (yy >= H // 4) & (yy < int(0.8 * H)), "bottom20": yy >= int(0.8 * H)}

    # 1. per-country region shares (true-class maps)
    rows = man[man.split == "test"].sample(frac=1, random_state=0)
    rows = rows[rows.groupby("label").cumcount() < 50].reset_index(drop=True)
    loader = make_loader(rows, reader, False, aug, masks, 64, 2)
    shares, labels, correct = [], [], []
    for x, y, _ in loader:
        cam, logits = grad_cam(model, prep(x.to(device)), target=y.to(device))
        up = F.interpolate(cam[:, None].float(), size=(H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        tot = up.sum((1, 2)) + 1e-8
        shares.append(np.stack([up[:, m].sum(1) / tot for m in regions.values()], 1))
        labels.append(y.numpy()); correct.append((logits.argmax(1).cpu().numpy() == y.numpy()))
    S = np.concatenate(shares); L = np.concatenate(labels)
    per = pd.DataFrame(S, columns=list(regions)).assign(country=[classes[i] for i in L])
    empty = per[list(regions)].sum(1) < 0.5          # no positive evidence for the true class at all
    print(f"images with an empty true-class map (excluded): {int(empty.sum())} of {len(per)}")
    per_c = per[~empty].groupby("country").mean()
    area = {k: float(v.mean()) for k, v in regions.items()}
    print("region areas:", area)
    print(per_c.sort_values("bottom20").round(3).to_string())

    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    order = per_c.sort_values("top25").index
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for k, col in (("top25", "#7fb3d5"), ("middle", "#a9a9a9"), ("bottom20", "#c0504d")):
        ax.bar(x, 100 * per_c.loc[order, k], bottom=bottom, color=col, width=0.8,
               label={"top25": "top quarter (sky, skyline)", "middle": "middle", "bottom20": "bottom fifth (road)"}[k])
        bottom += 100 * per_c.loc[order, k].to_numpy()
    for k, v in area.items():
        pass
    ax.set_xticks(x, order, fontsize=6.6)
    ax.set_ylabel("share of evidence (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_country_cam.{ext}")
    plt.close(fig)

    # 2. confusable pairs: images of A predicted as B, maps for both classes
    pairs = [("AT", "CH"), ("CA", "US")]
    pred = pd.read_csv("runs/ft_imagenet/eval/predictions.csv", dtype=str, keep_default_na=False)
    pick = []
    for a, b in pairs:
        cand = pred[(pred.split == "test") & (pred.country == a) & (pred.top3.str.contains(b))]
        cand = cand[cand.pred.isin([a, b])].drop_duplicates("creator_id").head(2)
        pick += [(r.id, a, b) for r in cand.itertuples()]
    prow = man.set_index("id").loc[[p[0] for p in pick]].reset_index()
    fig, axes = plt.subplots(len(pick), 3, figsize=(5.6, 1.45 * len(pick)))
    loader = make_loader(prow, reader, False, aug, masks, len(pick), 0)
    x, y, _ = next(iter(loader))
    xp = prep(x.to(device))
    maps = {}
    for cls in {c for _, a, b in pick for c in (a, b)}:
        t = torch.full((len(pick),), lab[cls], device=device)
        cam, _ = grad_cam(model, xp, target=t)
        maps[cls] = F.interpolate(cam[:, None].float(), size=(H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    with torch.no_grad():
        prob = torch.softmax(model(xp), 1).cpu().numpy()
    for i, (iid, a, b) in enumerate(pick):
        img = np.array(reader.image(iid))
        axes[i, 0].imshow(img); axes[i, 0].set_title(f"true {a}" if i == 0 else "", fontsize=7)
        for j, cls in ((1, a), (2, b)):
            axes[i, j].imshow(img); axes[i, j].imshow(maps[cls][i], cmap="jet", alpha=0.42)
            axes[i, j].set_title(f"evidence for {cls} (p={prob[i, lab[cls]]:.2f})", fontsize=6.6)
        axes[i, 0].set_ylabel(f"{a} vs {b}", fontsize=7)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_pairs.{ext}", dpi=200)
    plt.close(fig)

    # 3. confusion matrix (row-normalised), grouped by region
    cm = pd.read_csv("runs/ft_imagenet/eval/confusion_test.csv", index_col=0).loc[REGION_ORDER, REGION_ORDER]
    cmn = cm.div(cm.sum(1), axis=0).to_numpy() * 100
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=60)
    ax.set_xticks(range(32), REGION_ORDER, fontsize=5.6, rotation=90)
    ax.set_yticks(range(32), REGION_ORDER, fontsize=5.6)
    for k in (19, 24, 30):
        ax.axhline(k - 0.5, color="0.4", lw=0.6); ax.axvline(k - 0.5, color="0.4", lw=0.6)
    ax.set_xlabel("predicted country"); ax.set_ylabel("true country")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("% of true-country images (capped at 60)")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_confusion.{ext}")
    plt.close(fig)
    within = sum(cm.loc[a, b] for blk in [REGION_ORDER[:19], REGION_ORDER[19:24], REGION_ORDER[24:30]]
                 for a in blk for b in blk if a != b)
    errors = int(cm.to_numpy().sum() - np.trace(cm.to_numpy()))
    res = {"region_area": area, "empty_maps_excluded": int(empty.sum()), "sample": int(len(per)), "per_country_shares": per_c.round(4).to_dict(orient="index"),
           "pairs": [{"id": i, "true": a, "alt": b} for i, a, b in pick],
           "errors_within_block": int(within), "errors_total": errors}
    json.dump(res, open("results/final/country_features.json", "w"), indent=1)
    print("errors within Europe/Americas/Asia blocks:", within, "of", errors)


if __name__ == "__main__":
    main()
