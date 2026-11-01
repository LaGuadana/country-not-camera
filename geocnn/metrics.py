"""Accuracy metrics and contributor-clustered bootstrap confidence intervals.

Images of one contributor are not independent, so CIs resample contributors with replacement
(cluster bootstrap) rather than images.
"""
import numpy as np


def summarize(logits, labels, num_classes):
    pred = logits.argmax(1)
    top3 = np.argsort(-logits, axis=1)[:, :3]
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(conf, (labels, pred), 1)
    support = conf.sum(1)
    recall = np.divide(np.diag(conf), support, out=np.full(num_classes, np.nan), where=support > 0)
    return {
        "n": int(len(labels)),
        "top1": float((pred == labels).mean()),
        "top3": float((top3 == labels[:, None]).any(1).mean()),
        "balanced_acc": float(np.nanmean(recall)),
        "per_class_recall": recall.tolist(),
        "confusion": conf.tolist(),
    }


def cluster_bootstrap(correct, labels, clusters, num_classes, n_boot=1000, seed=0):
    """Bootstrap draws of top-1 and balanced accuracy, resampling clusters (contributors).
    Returns two arrays of length n_boot."""
    rng = np.random.default_rng(seed)
    codes, cluster_idx = np.unique(clusters, return_inverse=True)
    k = len(codes)
    onehot = np.zeros((len(labels), num_classes))
    onehot[np.arange(len(labels)), labels] = 1
    top1 = np.empty(n_boot)
    bacc = np.empty(n_boot)
    for start in range(0, n_boot, 100):
        b = min(100, n_boot - start)
        counts = rng.multinomial(k, np.full(k, 1.0 / k), size=b)  # (b, k) times each contributor drawn
        w = counts[:, cluster_idx].astype(np.float64)              # (b, n) weight of each image
        top1[start:start + b] = (w @ correct) / w.sum(1)
        hits = (w * correct) @ onehot
        tot = w @ onehot
        with np.errstate(invalid="ignore", divide="ignore"):
            bacc[start:start + b] = np.nanmean(hits / np.where(tot > 0, tot, np.nan), axis=1)
    return top1, bacc


def ci(draws, level=0.95):
    lo, hi = np.nanpercentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [float(lo), float(hi)]
