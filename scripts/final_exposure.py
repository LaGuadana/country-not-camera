"""Reconstruct distinct training images seen by each selected checkpoint.

Uses the actual seeded CoverageSampler and the best epoch in each checkpoint;
the earlier stage's best checkpoint, not its last epoch, initializes the next.
Computes country-adjusted exposure-bin differences and independent clustered
bootstrap intervals for both shared and disjoint groups.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geocnn.data import CoverageSampler, open_data  # noqa: E402
from scripts.train import chunk_rows  # noqa: E402

OUT = ROOT / "report/final_results"
RUNS = ("probe", "cpu_ft", "stage_1", "stage_2", "stage_3")
BINS = [(0, 1, "1"), (1, 6, "2-6"), (6, 21, "7-21"),
        (21, 51, "22-51"), (51, float("inf"), "52+")]
N_BOOT = 1000


def checkpoint(run):
    return torch.load(ROOT / "runs" / run / "best.pt", map_location="cpu", weights_only=False)


def count_table(rows):
    return rows.drop_duplicates("id").groupby("creator_id").size()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man, classes, _, _ = open_data(str(ROOT / "data/c32"))
    initial = man[man["split"] == "train"]
    probe_config = checkpoint("probe")["config"]["args"]
    shuffled = initial.sample(frac=1, random_state=probe_config["seed"])
    probe = shuffled[shuffled.groupby("label").cumcount() < probe_config["per_class"]]
    seen = {"probe": count_table(probe)}
    # The selected CPU checkpoint has traversed at least one full training
    # permutation; verify rather than silently assuming this remains true.
    cpu = checkpoint("cpu_ft")
    a = cpu["config"]["args"]
    consumed = (cpu["epoch"] + 1) * (a["epoch_size"] // a["batch_size"]) * a["batch_size"]
    if consumed < len(initial):
        raise RuntimeError("CPU checkpoint did not cover initial set; reconstruct sampler explicitly")
    accumulated = initial[["id", "creator_id"]].copy()
    seen["cpu_ft"] = count_table(accumulated)
    descriptions = {"probe": {"distinct_images": len(probe), "selected_epoch": 0},
                    "cpu_ft": {"distinct_images": len(accumulated), "selected_epoch": int(cpu["epoch"])}}
    for stage in range(1, 4):
        run = f"stage_{stage}"
        ck = checkpoint(run)
        a = ck["config"]["args"]
        rows = chunk_rows(str(ROOT / "data/c32"), man, stage, a["replay_floor"], a["seed"]).reset_index(drop=True)
        sampler = CoverageSampler(len(rows), a["epoch_size"], a["seed"])
        indices = []
        for epoch in range(ck["epoch"] + 1):
            sampler.set_epoch(epoch)
            idx = list(iter(sampler))
            # DataLoader drops the incomplete final batch.
            idx = idx[:len(idx) // a["batch_size"] * a["batch_size"]]
            indices.extend(idx)
        accumulated = pd.concat([accumulated, rows.iloc[indices][["id", "creator_id"]]],
                                ignore_index=True).drop_duplicates("id")
        seen[run] = count_table(accumulated)
        descriptions[run] = {"distinct_images": len(accumulated),
                             "selected_epoch": int(ck["epoch"]),
                             "epochs_seen_in_stage": int(ck["epoch"] + 1),
                             "stage_candidate_images": len(rows)}

    results = {}
    for run in RUNS:
        p = pd.read_csv(ROOT / "runs" / run / "eval/predictions.csv", dtype=str)
        p["correct"] = (p["country"] == p["pred"]).astype(float)
        test = p[p["split"] == "test"].copy()
        shared = p[p["split"] == "test_shared"].copy()
        shared["n_train"] = shared["creator_id"].map(seen[run]).fillna(0).astype(int)
        country_test = test.groupby("country")["correct"].mean().reindex(classes)
        # Bootstrap the reference test-country accuracy once per draw.
        codes, idx = np.unique(test["creator_id"], return_inverse=True)
        lab = pd.Categorical(test["country"], categories=classes).codes
        total = np.zeros((len(codes), len(classes)))
        hit = np.zeros_like(total)
        np.add.at(total, (idx, lab), 1)
        np.add.at(hit, (idx, lab), test["correct"].to_numpy())
        rng_test = np.random.default_rng(4400)
        ref_draw = np.empty((N_BOOT, len(classes)))
        for start in range(0, N_BOOT, 100):
            n = min(100, N_BOOT - start)
            w = rng_test.multinomial(len(codes), np.full(len(codes), 1 / len(codes)), size=n)
            denom = w @ total
            ref_draw[start:start+n] = np.divide(w @ hit, denom,
                out=np.full_like(denom, np.nan), where=denom > 0)
        rng_shared = np.random.default_rng(5500)
        bins = []
        for lo, hi, label in BINS:
            b = shared[(shared["n_train"] > lo) & (shared["n_train"] <= hi)]
            if b.empty:
                continue
            codes, idx = np.unique(b["creator_id"], return_inverse=True)
            lab = pd.Categorical(b["country"], categories=classes).codes
            n_by = np.zeros((len(codes), len(classes)))
            h_by = np.zeros_like(n_by)
            np.add.at(n_by, (idx, lab), 1)
            np.add.at(h_by, (idx, lab), b["correct"].to_numpy())
            excess = float((b["correct"].to_numpy() - country_test.to_numpy()[lab]).mean())
            draws = np.empty(N_BOOT)
            for start in range(0, N_BOOT, 100):
                n = min(100, N_BOOT - start)
                w = rng_shared.multinomial(len(codes), np.full(len(codes), 1 / len(codes)), size=n)
                n_country, h_country = w @ n_by, w @ h_by
                difference = h_country - n_country * ref_draw[start:start+n]
                draws[start:start+n] = np.nansum(difference, axis=1) / n_country.sum(axis=1)
            bins.append({"bin": label, "images": len(b), "contributors": len(codes),
                         "accuracy": float(b["correct"].mean()), "adjusted_excess": excess,
                         "excess_ci95": np.percentile(draws, [2.5, 97.5]).tolist()})
        results[run] = {"checkpoint": descriptions[run], "bins": bins,
                        "shared_without_training_images": int((shared["n_train"] == 0).sum())}
    (OUT / "exposure.json").write_text(json.dumps(
        {"bootstrap_draws": N_BOOT, "runs": results}, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(6.8, 3.5), layout="constrained")
    x = np.arange(4)
    for run, shift, color, label in (
            ("cpu_ft", -0.08, "#255c85", "Partial fine-tune"),
            ("stage_3", 0.08, "#bd5548", "Stage 3")):
        by_bin = {b["bin"]: b for b in results[run]["bins"]}
        selected = [by_bin[k] for k in ("2-6", "7-21", "22-51", "52+")]
        y = np.array([b["adjusted_excess"] for b in selected]) * 100
        ci = np.array([b["excess_ci95"] for b in selected]) * 100
        ax.errorbar(x + shift, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="o-",
                    capsize=3, color=color, label=label)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(xticks=x, xticklabels=("2–6", "7–21", "22–51", "52+"),
           xlabel="Distinct training images of the contributor",
           ylabel="Country-adjusted excess accuracy (pp)")
    ax.legend()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"exposure.{ext}", dpi=200)
    plt.close(fig)
    for run, r in results.items():
        print(run, r["checkpoint"], "zero exposure:", r["shared_without_training_images"])
        for b in r["bins"]:
            print(" ", b["bin"], "n", b["images"], "excess", round(b["adjusted_excess"] * 100, 2),
                  "CI", [round(x * 100, 2) for x in b["excess_ci95"]])


if __name__ == "__main__":
    main()
