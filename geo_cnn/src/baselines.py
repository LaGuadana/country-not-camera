"""First baselines on the balanced 10-country subset.

Models: majority class (train-set mode), uniform chance, multinomial logistic regression
on HSV histograms, and (if the feature file exists) logistic regression on frozen
ResNet-18 features. C is chosen by 3-fold StratifiedGroupKFold on the training part only.
Reports top-1 / top-3 accuracy with Wilson 95 % CIs and saves the confusion matrix of the
best model.

Usage:
  python -I src/baselines.py --split data/processed/split.csv \
      --feat hsv=data/processed/feat_hsv.npz --feat resnet18=data/processed/feat_resnet18.npz \
      --results results --figdir figures
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402
from countries import name_of  # noqa: E402


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def fit_lr(Xtr, ytr, gtr, sqrt, seed):
    steps = ([FunctionTransformer(np.sqrt)] if sqrt else []) + [
        StandardScaler(), LogisticRegression(max_iter=3000)]
    pipe = make_pipeline(*steps)
    grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1.0]}
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=1)
    gs.fit(Xtr, ytr, groups=gtr)
    return gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--feat", action="append", default=[], help="name=path.npz")
    ap.add_argument("--results", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _strict")
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--split-label", default="split: sequence- and 0.1° cell-disjoint")
    a = ap.parse_args()
    style.apply()

    df = pd.read_csv(a.split, dtype={"id": str, "country": str}, keep_default_na=False)
    classes = sorted(df["country"].unique())
    cidx = {c: i for i, c in enumerate(classes)}
    y = df["country"].map(cidx).values
    tr = (df["split"] == "train").values
    te = ~tr
    g = df["group"].values
    n_te = int(te.sum())
    K = len(classes)
    res = {"classes": classes, "n_train": int(tr.sum()), "n_test": n_te, "models": {},
           "test_class_counts": {classes[k]: int((y[te] == k).sum()) for k in range(K)}}

    # majority class: mode of the training labels (ties broken by first index)
    maj = np.bincount(y[tr], minlength=K).argmax()
    order = np.argsort(-np.bincount(y[tr], minlength=K), kind="stable")
    top1 = float((y[te] == maj).mean())
    top3 = float(np.isin(y[te], order[:3]).mean())
    res["models"]["majority"] = {"top1": top1, "top3": top3,
                                 "top1_ci95": wilson(round(top1 * n_te), n_te), "balanced_top1": 1 / K}
    res["models"]["uniform_chance_expected"] = {"top1": 1 / K, "top3": 3 / K}

    best, best_name = None, None
    for spec in a.feat:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print("missing", path, "- skipped")
            continue
        z = np.load(path, allow_pickle=False)
        f = pd.DataFrame(z["X"], index=z["ids"].astype(str))
        X = f.loc[df["id"].values].values.astype(np.float64)
        gs = fit_lr(X[tr], y[tr], g[tr], sqrt=(name == "hsv"), seed=a.seed)
        P = gs.predict_proba(X[te])
        pred = P.argmax(1)
        t1 = float((pred == y[te]).mean())
        t3 = float(top_k_accuracy_score(y[te], P, k=3, labels=np.arange(K)))
        res["models"][f"logreg_{name}"] = {
            "top1": t1, "top3": t3, "top1_ci95": wilson(int((pred == y[te]).sum()), n_te),
            "best_C": gs.best_params_["logisticregression__C"],
            "cv_top1_train": float(gs.best_score_), "dim": int(X.shape[1]),
            "per_class_recall": {classes[k]: float((pred[y[te] == k] == k).mean()) for k in range(K)},
        }
        r = res["models"][f"logreg_{name}"]
        r["balanced_top1"] = float(np.mean(list(r["per_class_recall"].values())))
        print(name, res["models"][f"logreg_{name}"], flush=True)
        if best is None or t1 > best[0]:
            best, best_name = (t1, pred), name

    with open(os.path.join(a.results, f"baseline_results{a.tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))

    if best is not None and not a.no_fig:
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y[te], best[1], labels=np.arange(K))
        cmn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        im = ax.imshow(cmn * 100, cmap="Blues", vmin=0, vmax=100)
        ax.grid(False)
        labs = [f"{name_of(c)}" for c in classes]
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(labs, rotation=45, ha="right")
        ax.set_yticklabels(labs)
        for i in range(K):
            for j in range(K):
                v = cmn[i, j] * 100
                if cm[i, j] > 0:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 55 else style.INK)
        ax.tick_params(length=0)
        ax.set_xlabel("predicted country")
        ax.set_ylabel("true country")
        m = res["models"][f"logreg_{best_name}"]
        title = {"hsv": "HSV histogram", "resnet18": "frozen ResNet-18"}.get(best_name, best_name)
        ax.set_title(f"Confusion matrix (row %), {title} + log. reg.\n"
                     f"top-1 {100 * m['top1']:.1f}% on n = {n_te} held-out images\n{a.split_label}", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("% of true-country images")
        cb.outline.set_visible(False)
        style.save(fig, a.figdir, f"fig_e_confusion_best{a.tag}")


if __name__ == "__main__":
    main()
