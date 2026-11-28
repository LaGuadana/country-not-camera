"""Recompute report metrics from saved predictions without changing existing results.

Usage: OPENBLAS_NUM_THREADS=1 python -I scripts/final_results.py
Resamples contributors independently between splits, and uses the same resamples for
all models within a split. Outputs go to report/final_results/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "final_results"
RUNS = ("classical", "probe", "cpu_ft", "stage_1", "stage_2", "stage_3")
N_BOOT = 2000


def load(run):
    p = pd.read_csv(ROOT / "runs" / run / "eval" / "predictions.csv",
                    dtype=str, keep_default_na=False)
    p = p[p["split"].isin(["test", "test_shared"])].copy()
    if p.duplicated(["split", "id"]).any():
        raise ValueError(f"Duplicate image IDs in {run}")
    return p.set_index(["split", "id"]).sort_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    classes = json.loads((ROOT / "data/c32/classes.json").read_text())
    data = {r: load(r) for r in RUNS}
    ref = data["stage_3"]
    for name, p in data.items():
        if not p.index.equals(ref.index):
            raise ValueError(f"Different evaluation IDs in {name}")
        if not p[["country", "creator_id"]].equals(ref[["country", "creator_id"]]):
            raise ValueError(f"Different labels or contributors in {name}")

    point, draws, recall = {}, {}, {}
    for j, split in enumerate(("test", "test_shared")):
        base = ref.xs(split)
        labels = pd.Categorical(base["country"], categories=classes).codes
        if (labels < 0).any():
            raise ValueError("Unknown country")
        _, cluster = np.unique(base["creator_id"], return_inverse=True)
        n_clusters = cluster.max() + 1
        # Per contributor and class counts let each bootstrap draw score all models
        # without allocating a bootstrap-by-image matrix.
        totals = np.zeros((n_clusters, len(classes)), dtype=np.int64)
        np.add.at(totals, (cluster, labels), 1)
        hits = []
        for run in RUNS:
            p = data[run].xs(split)
            correct = (p["pred"].to_numpy() == base["country"].to_numpy())
            h = np.zeros_like(totals)
            np.add.at(h, (cluster, labels), correct.astype(int))
            hits.append(h)
            denom = totals.sum(axis=0)
            rec = h.sum(axis=0) / denom
            point[(run, split)] = float(rec.mean())
            recall[(run, split)] = rec
        hits = np.stack(hits)
        rng = np.random.default_rng(20261125 + j)
        results = np.empty((N_BOOT, len(RUNS)))
        for start in range(0, N_BOOT, 100):
            size = min(100, N_BOOT - start)
            weights = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=size)
            denom = weights @ totals
            for k in range(len(RUNS)):
                num = weights @ hits[k]
                rates = np.divide(num, denom, out=np.full_like(num, np.nan, dtype=float),
                                  where=denom > 0)
                results[start:start + size, k] = np.nanmean(rates, axis=1)
        draws[split] = results

    def interval(x):
        return np.percentile(x, [2.5, 97.5]).tolist()

    rows = []
    for k, run in enumerate(RUNS):
        test, shared = point[run, "test"], point[run, "test_shared"]
        gap = draws["test_shared"][:, k] - draws["test"][:, k]
        base_index = RUNS.index("cpu_ft")
        test_change = draws["test"][:, k] - draws["test"][:, base_index]
        gap_change = gap - (draws["test_shared"][:, base_index] -
                            draws["test"][:, base_index])
        row = {"model": run, "test_bacc": test, "test_ci": interval(draws["test"][:, k]),
               "shared_bacc": shared, "shared_ci": interval(draws["test_shared"][:, k]),
               "gap": shared - test, "gap_ci": interval(gap),
               "test_change_vs_cpu_ft": test - point["cpu_ft", "test"],
               "test_change_ci": interval(test_change),
               "gap_change_vs_cpu_ft": (shared - test) -
                                        (point["cpu_ft", "test_shared"] - point["cpu_ft", "test"]),
               "gap_change_ci": interval(gap_change)}
        rows.append(row)
    (OUT / "metrics.json").write_text(json.dumps(
        {"bootstrap_draws": N_BOOT, "seed_test": 20261125, "seed_shared": 20261126,
         "method": "independent contributor resampling between splits; paired across models",
         "results": rows}, indent=2) + "\n")
    pd.DataFrame([{"country": c, **{r: recall[r, "test"][i] for r in RUNS}}
                  for i, c in enumerate(classes)]).to_csv(OUT / "country_recall.csv", index=False)

    names = ["Classical", "Linear probe", "Partial fine-tune", "Stage 1", "Stage 2", "Stage 3"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    y = np.arange(len(RUNS))
    for split, offset, color, label in [("test", -0.15, "#255c85", "Disjoint contributors"),
                                        ("test_shared", 0.15, "#bd5548", "Training-pool contributors")]:
        values = np.array([point[r, split] for r in RUNS]) * 100
        bounds = np.array([interval(draws[split][:, k]) for k in range(len(RUNS))]) * 100
        axes[0].errorbar(values, y + offset, xerr=[values - bounds[:, 0], bounds[:, 1] - values],
                         fmt="o", capsize=3, color=color, label=label)
    axes[0].set(yticks=y, yticklabels=names, xlabel="Balanced accuracy (%)", title="Model comparison")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc="upper right")
    gap = np.array([r["gap"] for r in rows]) * 100
    bounds = np.array([r["gap_ci"] for r in rows]) * 100
    axes[1].errorbar(gap, y, xerr=[gap - bounds[:, 0], bounds[:, 1] - gap],
                     fmt="o", capsize=3, color="#604391")
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[1].set(yticks=y, yticklabels=names, xlabel="Shared − disjoint (percentage points)",
                title="Contributor gap")
    axes[1].invert_yaxis()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"model_comparison.{ext}", dpi=200)
    plt.close(fig)

    order = np.argsort(recall["stage_3", "test"])
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.barh(np.arange(len(order)), 100 * recall["stage_3", "test"][order], color="#255c85")
    ax.set(yticks=np.arange(len(order)), yticklabels=np.array(classes)[order],
           xlabel="Recall on unseen contributors (%)", title="Stage 3: recall by country")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"country_recall.{ext}", dpi=200)
    plt.close(fig)
    for r in rows:
        print(f'{r["model"]:10} test {100*r["test_bacc"]:.2f}% shared '
              f'{100*r["shared_bacc"]:.2f}% gap {100*r["gap"]:+.2f} pp '
              f'CI [{100*r["gap_ci"][0]:+.2f}, {100*r["gap_ci"][1]:+.2f}]')
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
