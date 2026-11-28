"""Top-1/2/3 accuracy, border-tolerant accuracy and per-country distinguishability maps.

Three steps. `probs` needs torch (project venv); `accuracy` and `maps` need geopandas/shapely
(the project venv has neither; any venv with geopandas >= 1.0 works):
  python -I scripts/topk_border_similarity.py probs      # project venv
  python    scripts/topk_border_similarity.py accuracy   # geopandas venv
  python    scripts/topk_border_similarity.py maps       # geopandas venv

probs     Probabilities recomputed from the saved 512-d features and the checkpoint's linear
          head (eval mode, dropout off); checked against evaluate.py's predictions.
accuracy  top-1/2/3 (plain and balanced); border-tolerant top-1: a wrong guess counts if the
          photo lies within X km of the predicted country; accuracy by distance of the photo
          to its own country's border.
maps      Pairwise distinguishability of every pair of countries on the contributor-disjoint
          test split: among photos of A or B, how often the model gives the true one of the
          two the higher probability (balanced over A and B; 50 % = coin flip, 100 % = always
          separated). One world map per country plus a clustered matrix.

Border data: Natural Earth 1:50m admin-0 countries (public domain), data/borders/.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

OUT = "results/topk_border"
SPLITS = ["test", "test_shared"]
EARTH_R = 6371.0
BORDERS = "data/borders/ne_50m_admin_0_countries.geojson"
TOLERANCE_KM = [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000]

NAMES = {"AT": "Austria", "AU": "Australia", "BE": "Belgium", "BR": "Brazil", "CA": "Canada",
         "CH": "Switzerland", "CO": "Colombia", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark",
         "ES": "Spain", "ET": "Ethiopia", "FR": "France", "GB": "United Kingdom", "GR": "Greece",
         "HR": "Croatia", "HU": "Hungary", "ID": "Indonesia", "IN": "India", "IT": "Italy",
         "JP": "Japan", "MX": "Mexico", "MY": "Malaysia", "NL": "Netherlands", "PL": "Poland",
         "PT": "Portugal", "RU": "Russia", "SE": "Sweden", "TH": "Thailand", "TR": "Turkey",
         "TW": "Taiwan", "US": "United States"}


def load_probs(split):
    z = np.load(os.path.join(OUT, f"probs_{split}.npz"))
    return z["p"], z["labels"], z["ids"], [str(c) for c in z["classes"]]


# --------------------------------------------------------------------------- probs
def probs(a):
    import torch
    ck = torch.load(os.path.join(a.run, "best.pt"), map_location="cpu", weights_only=False)
    W, b = ck["model"]["head.weight"].numpy(), ck["model"]["head.bias"].numpy()
    cls = ck["config"]["classes"]
    pred = pd.read_csv(os.path.join(a.run, "eval", "predictions.csv"), dtype=str, keep_default_na=False)
    os.makedirs(OUT, exist_ok=True)
    for s in SPLITS:
        z = np.load(os.path.join(a.run, "eval", f"features_{s}.npz"))
        lg = z["features"] @ W.T + b
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        ps = pred[pred["split"] == s]
        assert (ps["id"].to_numpy() == z["ids"]).all()
        agree = np.mean(np.array(cls)[p.argmax(1)] == ps["pred"].to_numpy())
        print(f"{s}: n={len(p)}, argmax agrees with evaluate.py predictions: {agree:.4f}")
        assert agree > 0.999
        np.savez_compressed(os.path.join(OUT, f"probs_{s}.npz"), p=p.astype(np.float32),
                            labels=z["labels"], ids=z["ids"], classes=np.array(cls))


# --------------------------------------------------------------------------- accuracy
def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def chord_to_km(c):
    return 2 * EARTH_R * np.arcsin(np.clip(c / 2, 0, 1))


def country_geoms(world, cls):
    # ISO_A2 is -99 for France and Norway in Natural Earth; ISO_A2_EH is filled in.
    # Some countries come as several features (Australia + its island territories): union them.
    out = {}
    for c in cls:
        g = world.loc[world["ISO_A2_EH"] == c, "geometry"]
        assert len(g) >= 1, c
        out[c] = g.union_all()
    return out


def accuracy(a):
    import geopandas as gpd
    import shapely
    from scipy.spatial import cKDTree

    man = pd.read_csv("data/c32/manifest.csv", dtype={"id": str}, keep_default_na=False).set_index("id")
    cls = load_probs("test")[3]
    geoms = country_geoms(gpd.read_file(BORDERS), cls)
    trees = {}
    for c, g in geoms.items():
        pts = shapely.get_coordinates(shapely.segmentize(g.boundary, 0.02))  # a vertex every ~2 km
        trees[c] = cKDTree(xyz(pts[:, 1], pts[:, 0]))

    res, tol_rows = {}, []
    for s in SPLITS:
        P, y, ids, _ = load_probs(s)
        rank = np.argsort(-P, 1)
        r = {"n": int(len(y))}
        for k in (1, 2, 3):
            h = (rank[:, :k] == y[:, None]).any(1)
            r[f"top{k}"] = float(h.mean())
            r[f"balanced_top{k}"] = float(np.mean([h[y == c].mean() for c in range(len(cls))]))

        lat = man.loc[ids, "latitude"].astype(float).to_numpy()
        lon = man.loc[ids, "longitude"].astype(float).to_numpy()
        pred = rank[:, 0]
        correct = pred == y
        d_pred = np.zeros(len(y))     # km from the photo to the predicted country (0 if inside)
        d_own = np.zeros(len(y))      # km from the photo to its own country's border line
        in_own = np.zeros(len(y), bool)
        for k, c in enumerate(cls):
            m = pred == k
            if m.any():
                inside = shapely.contains_xy(geoms[c], lon[m], lat[m])
                d = chord_to_km(trees[c].query(xyz(lat[m], lon[m]))[0])
                d_pred[m] = np.where(inside, 0.0, d)
            m = y == k
            d_own[m] = chord_to_km(trees[c].query(xyz(lat[m], lon[m]))[0])
            in_own[m] = shapely.contains_xy(geoms[c], lon[m], lat[m])
        d_pred[correct] = 0.0

        wrong = ~correct
        for t in TOLERANCE_KM:
            near = wrong & (d_pred <= t)
            tol_rows.append({"split": s, "tolerance_km": t,
                             "accuracy": float((correct | near).mean()),
                             "gain_pp": float(100 * near.mean()),
                             "errors_forgiven": float(near.sum() / wrong.sum())})
        r["median_km_wrong_to_predicted_country"] = float(np.median(d_pred[wrong]))
        r["frac_images_outside_own_polygon"] = float((~in_own).mean())
        bins = [0, 10, 25, 50, 100, 250, np.inf]
        labels = ["<10", "10-25", "25-50", "50-100", "100-250", ">250"]
        r["accuracy_by_km_to_own_border"] = {
            n: {"n": int(m.sum()), "top1": float(correct[m].mean())}
            for n, lo, hi in zip(labels, bins[:-1], bins[1:]) for m in [(d_own >= lo) & (d_own < hi)]}
        res[s] = r
        pd.DataFrame({"id": ids, "true": np.array(cls)[y], "pred": np.array(cls)[pred],
                      "km_to_predicted_country": d_pred, "km_to_own_border": d_own,
                      "inside_own_polygon": in_own}).to_csv(os.path.join(OUT, f"per_image_{s}.csv"), index=False)

    tol = pd.DataFrame(tol_rows)
    tol.to_csv(os.path.join(OUT, "border_tolerance.csv"), index=False)
    with open(os.path.join(OUT, "accuracy.json"), "w") as f:
        json.dump(res, f, indent=1)
    wide = tol.pivot(index="tolerance_km", columns="split", values="accuracy")[SPLITS]
    print((100 * wide).round(1).to_string())
    for s in SPLITS:
        print(s, {k: round(v, 3) for k, v in res[s].items() if k.startswith(("top", "balanced"))})


# --------------------------------------------------------------------------- maps
def pairwise_distinguishability(P, y, K):
    """D[i, j]: among photos of i or j, how often the model gives the true one of the two the
    higher probability; the two directions are weighted equally."""
    D = np.full((K, K), np.nan)
    for i in range(K):
        a = P[y == i]
        for j in range(K):
            if i != j:
                b = P[y == j]
                D[i, j] = 0.5 * ((a[:, i] > a[:, j]).mean() + (b[:, j] > b[:, i]).mean())
    return D


# Classes of pairwise accuracy; one blue ramp, darker = harder to tell apart.
BIN_EDGES = [0.50, 0.65, 0.75, 0.85, 0.92, 1.001]
BIN_LABELS = ["under 65 % (near coin flip)", "65–75 %", "75–85 %", "85–92 %", "92 % or more (easy)"]
BIN_COLORS = ["#0e3d7e", "#1b64be", "#338ced", "#84b9f3", "#cce3fd"]
FOCUS = "#f47230"          # the selected country
OTHER = "#e4e4e4"          # countries outside the 32
EDGE = "#ffffff"
INK, INK2 = "#191919", "#666666"
EUROPE = (2.55e6, 7.45e6, 1.35e6, 5.45e6)   # LAEA Europe (EPSG:3035) window


def maps(a):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    P, y, _, cls = load_probs("test")
    K = len(cls)
    D = pairwise_distinguishability(P, y, K)
    pd.DataFrame(D, index=cls, columns=cls).round(4).to_csv(os.path.join(OUT, "distinguishability_matrix.csv"))
    recall = np.array([(P[y == i].argmax(1) == i).mean() for i in range(K)])
    order = np.argsort(np.where(np.isnan(D), np.inf, D), 1)
    pd.DataFrame({
        "country": cls, "name": [NAMES[c] for c in cls], "recall": recall.round(4),
        "hardest_to_tell_apart_from": [NAMES[cls[order[i, 0]]] for i in range(K)],
        "pairwise_accuracy_vs_hardest": [round(D[i, order[i, 0]], 4) for i in range(K)],
        "mean_pairwise_accuracy": np.nanmean(D, 1).round(4),
    }).to_csv(os.path.join(OUT, "country_summary.csv"), index=False)

    world = gpd.read_file(BORDERS)
    world = world[world["CONTINENT"] != "Antarctica"].copy()
    world["iso"] = world["ISO_A2_EH"]
    world = world.dissolve("iso", as_index=False)[["iso", "geometry"]]
    w_world = world.to_crs("EPSG:8857")                     # Equal Earth
    w_world["geometry"] = w_world.simplify(6000)
    w_eu = world.to_crs("EPSG:3035")                        # Europe, equal area
    w_eu = w_eu[w_eu.intersects(gpd.GeoSeries.from_xy([EUROPE[0], EUROPE[1]], [EUROPE[2], EUROPE[3]],
                                                       crs=3035).union_all().envelope.buffer(1e6))].copy()
    w_eu["geometry"] = w_eu.simplify(2500)

    def colours(i):
        col = {}
        for j, c in enumerate(cls):
            col[c] = FOCUS if j == i else BIN_COLORS[np.searchsorted(BIN_EDGES, D[i, j], side="right") - 1]
        return [col.get(c, OTHER) for c in w_world["iso"]], [col.get(c, OTHER) for c in w_eu["iso"]]

    WORLD = (-1.66e7, 1.74e7, -6.6e6, 8.4e6)          # Equal Earth window without Antarctica
    aw = (WORLD[1] - WORLD[0]) / (WORLD[3] - WORLD[2])  # width / height of each panel
    ae = (EUROPE[1] - EUROPE[0]) / (EUROPE[3] - EUROPE[2])
    GAP = 0.06

    def draw(fig, rect, i, small):
        """World map with a Europe panel to its right (19 of the 32 countries are European).
        rect is in figure fractions and should have physical aspect (aw + GAP + ae) : 1."""
        cw, ce = colours(i)
        lw = 0.15 if small else 0.3
        x0, y0, w, h = rect
        tot = aw + GAP + ae
        ax = fig.add_axes([x0, y0, w * aw / tot, h])
        w_world.plot(ax=ax, color=cw, edgecolor=EDGE, linewidth=lw)
        ax.set_xlim(*WORLD[:2])
        ax.set_ylim(*WORLD[2:])
        ax.set_axis_off()
        eu = fig.add_axes([x0 + w * (aw + GAP) / tot, y0, w * ae / tot, h])
        w_eu.plot(ax=eu, color=ce, edgecolor=EDGE, linewidth=lw)
        eu.set_xlim(*EUROPE[:2])
        eu.set_ylim(*EUROPE[2:])
        eu.set_xticks([])
        eu.set_yticks([])
        eu.set_xlabel("")
        eu.set_ylabel("")
        for sp in eu.spines.values():
            sp.set_edgecolor("#c8c8c8")
            sp.set_linewidth(0.6)
        eu.text(0.03, 0.97, "Europe", transform=eu.transAxes, va="top",
                fontsize=6.5 if small else 9, color=INK2)
        return ax

    def map_rect(fig, x0, y0, width):
        """Rect of the given figure-fraction width whose height keeps the maps undistorted."""
        h_in = width * fig.get_figwidth() / (aw + GAP + ae)
        return [x0, y0, width, h_in / fig.get_figheight()]

    def hardest_text(i, n):
        return ", ".join(f"{NAMES[cls[j]]} {100 * D[i, j]:.0f} %" for j in order[i, :n])

    legend = ([Patch(facecolor=FOCUS, edgecolor="none", label="selected country")]
              + [Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(BIN_COLORS, BIN_LABELS)]
              + [Patch(facecolor=OTHER, edgecolor="none", label="not among the 32 countries")])

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)
    explain = ("Every other country is shaded by pairwise accuracy: among photos of the two countries, how often "
               "the model gives the right one of the two the higher probability.\n50 % is a coin flip; darker means "
               "harder to tell apart. Contributor-disjoint test split (12,800 photos), stage-3 model.")
    for i, c in enumerate(cls):
        fig = plt.figure(figsize=(12, 5.4))
        r = map_rect(fig, 0.015, 0.0, 0.97)
        r[1] = 0.135
        draw(fig, r, i, small=False)
        fig.text(0.015, 0.975, f"{NAMES[c]}: which countries the model confuses it with",
                 fontsize=15, weight="bold", va="top")
        fig.text(0.015, 0.905, f"Recognised correctly in {100 * recall[i]:.0f} % of test photos.  "
                 f"Hardest to tell apart from: {hardest_text(i, 3)}.", fontsize=10.5, color=INK, va="top")
        fig.text(0.015, 0.855, explain, fontsize=8.5, color=INK2, va="top", linespacing=1.4)
        fig.legend(handles=legend, loc="lower center", ncol=7, frameon=False, fontsize=8.5,
                   handlelength=1.4, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
        fig.savefig(os.path.join(OUT, "maps", f"{c}.png"), dpi=150, facecolor="white")
        plt.close(fig)

    # overview: all 32 maps
    ncol, nrow = 4, 8
    fig = plt.figure(figsize=(18, 17.5))
    top, bottom = 0.925, 0.035
    ph, pw = (top - bottom) / nrow, 1 / ncol
    for i, c in enumerate(cls):
        r_, k_ = divmod(i, ncol)
        x0, y0 = k_ * pw, top - (r_ + 1) * ph
        draw(fig, map_rect(fig, x0 + 0.008, y0 + 0.006, pw - 0.016), i, small=True)
        fig.text(x0 + 0.008, y0 + ph - 0.004, NAMES[c], fontsize=11.5, weight="bold", va="top")
        fig.text(x0 + 0.008, y0 + ph - 0.019, f"recall {100 * recall[i]:.0f} %  ·  closest: {hardest_text(i, 1)}",
                 fontsize=8.5, color=INK2, va="top")
    fig.text(0.008, 0.993, "Which countries the model confuses with each other", fontsize=18, weight="bold", va="top")
    fig.text(0.008, 0.973, "Selected country in orange. " + explain, fontsize=10, color=INK2, va="top",
             linespacing=1.4)
    fig.legend(handles=legend, loc="lower center", ncol=7, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(OUT, "distinguishability_maps_all.png"), dpi=110, facecolor="white")
    plt.close(fig)

    # clustered matrix of the same numbers
    S = 1 - np.nan_to_num(D, nan=1.0)
    np.fill_diagonal(S, 0)
    idx = leaves_list(linkage(squareform((S + S.T) / 2, checks=False), "average"))
    M = 100 * D[np.ix_(idx, idx)]
    from matplotlib.colors import BoundaryNorm, ListedColormap
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(M, cmap=ListedColormap(BIN_COLORS), norm=BoundaryNorm([100 * e for e in BIN_EDGES], 5))
    for (r_, k_), v in np.ndenumerate(M):
        if not np.isnan(v):
            ax.text(k_, r_, f"{v:.0f}", ha="center", va="center", fontsize=6.2,
                    color="white" if v < 85 else INK)
    lab = [NAMES[cls[j]] for j in idx]
    ax.set_xticks(range(K), lab, rotation=90, fontsize=8.5)
    ax.set_yticks(range(K), lab, fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, K), minor=True)
    ax.set_yticks(np.arange(-0.5, K), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[100 * e for e in BIN_EDGES[:-1]] + [100])
    cb.set_label("pairwise accuracy (%): darker = harder to tell apart", fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title("Pairwise accuracy between countries, clustered so similar countries sit together",
                 fontsize=12, weight="bold", loc="left", pad=10)
    fig.savefig(os.path.join(OUT, "distinguishability_matrix.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["probs", "accuracy", "maps"])
    ap.add_argument("--run", default="runs/stage_3")
    a = ap.parse_args()
    {"probs": probs, "accuracy": accuracy, "maps": maps}[a.step](a)
