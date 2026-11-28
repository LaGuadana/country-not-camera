# Results handoff — checked 25 November 2026

Working notes for the team, not submission-ready prose. The course brief prohibits
AI-generated submitted prose. The authors must write the paper, independently
check every claim, and disclose permitted AI assistance.

## Corrected model comparison

All values are balanced accuracy (%); the gap is shared minus disjoint in percentage
points. Intervals are 95% contributor-clustered bootstrap intervals, with independent
resampling between test sets and paired resampling across models, 2,000 draws.

| Model | Disjoint test [95% CI] | Shared test | Gap [95% CI] |
|---|---:|---:|---:|
| Classical HSV/HOG + PCA + kNN | 5.62 [5.08, 6.18] | 6.48 | 0.85 [0.01, 1.75] |
| Frozen ImageNet linear probe | 13.52 [12.77, 14.28] | 13.74 | 0.21 [-1.06, 1.47] |
| Partial fine-tune | 28.69 [27.55, 29.86] | 34.46 | 5.77 [4.00, 7.46] |
| Stage 1 | 29.81 [28.65, 30.97] | 36.81 | 7.00 [5.21, 8.81] |
| Stage 2 | 30.11 [28.97, 31.27] | 37.69 | 7.58 [5.81, 9.33] |
| Stage 3 | 30.66 [29.54, 31.80] | 38.01 | 7.35 [5.55, 9.12] |

- Stage 3 vs partial fine-tune, disjoint test: **+1.97 pp** [1.38, 2.54].
- Change in gap: **+1.58 pp** [0.66, 2.48].
- Stage 3 top-3 disjoint test: **52.41%**.
- Chance over 32 classes: **3.125%**.
- Test is balanced (400 images per country); shared test is not perfectly balanced.
  Therefore shared top-1 (37.62% for stage 3) and balanced accuracy (38.01%)
  are different. Do not substitute one for the other.
- Exact values, both split intervals and paired changes: `metrics.json`.
- `model_comparison.pdf` and `country_recall.pdf` are figures based on these results.

## Critical corrections to earlier README/progress report

1. **Actual training exposure:** reconstructed distinct training images for selected
   checkpoints: probe 9,600; partial fine-tune 101,578; stage 1 188,768;
   stage 2 **258,677**; stage 3 **346,197**. Stage 2 selected epoch 4 of 5
   (zero-based checkpoint epoch 3); stage 3 initializes from that checkpoint.
   Thus 275,962 and 363,150 are available cumulative images, not the number
   actually seen by the respective selected checkpoints. Reconstruction uses the
   saved seed, selected checkpoint epoch, replay rows and `CoverageSampler`.
2. **Baseline comparison is not a representation-only experiment:** classical
   kNN and logistic-regression probe have different classifiers and different
   training samples (938 of 9,600 images overlap). The classical script selects
   manifest order; the probe uses a seeded shuffled selection.
3. **“Shared” is relative to the full training pool**, not necessarily a smaller
   baseline's actual sample. Among 12,559 shared-test images, **2,030** are from
   contributors absent from the probe's sampled training data and **11,330** from
   contributors absent from the classical gallery. Do not describe those sets as
   entirely made up of photographers seen by the baseline.
4. **Exposure plot:** use `exposure.pdf` and `exposure.json`, not the old progress
   figure's cross-model exposure chart. The old script bins all models using the
   complete extra-data list, including images they never saw. For stage 3, the 52+
   bin has 10,536 images and +8.89 pp country-adjusted excess [7.02, 10.74].
   Bins differ in contributor composition; this is observational, not a causal
   dose-response. The corrected probe's 52+ bin contains only **one contributor**;
   don't draw a conclusion from it. Unexposed shared contributors are excluded
   from the positive-exposure bins, and their count is recorded in `exposure.json`.
5. **One pass, not five:** each extra-data stage divides approximately one pass
   into five epochs. The selected stage-2 checkpoint only completed four epochs.
6. **Retrieval analysis:** `runs/knn_probe.json` has an incorrect `n_query: 4`
   field (NPZ member count, not query-image count); `scripts/knn_probe.py` has
   been corrected. Its 4.7 pp conditional contrast is descriptive and does not
   establish that same-contributor retrieval caused that share of the gap.
7. **Scope of claims:** no completed matched fingerprint-removal training run or
   from-scratch model was found. The completed CNN is partially fine-tuned, not
   end-to-end. More-data runs also changed class weighting and use replay; a causal
   data-only claim is not justified.

## Data and evaluation checks

- Manifest: 101,578 initial train images; 6,400 disjoint validation; 12,800
  disjoint test; 6,359 shared validation; 12,559 shared test.
- Checked: zero contributor overlap between train/val/test, and no nonempty
  sequence overlap between train and any of those or the shared sets.
- Disjoint test: 3,043 contributors; shared test: 3,037 training-pool contributors.
- 4:3 center crop, 128×170 pixels. All images are drawn from the source dataset's
  train partition; these are project-defined evaluation splits.
- Best checkpoint chosen on a fixed 1,600-image validation subsample. Bootstrap
  intervals account for contributor clustering, not retraining variability,
  selection, or alternative contributor splits. One seed/split was used.
- For stage 3, highest per-class test recalls: ET 68.50%, IN 62.00%, TW 60.50%;
  lowest: FR 8.25%, DE 9.50%, CZ 12.75%. Country recalls have uncertainty and
  should not be treated as an exhaustive or representative country ranking.
- Shared/disjoint differences may also reflect location, route, contributor and
  sampling differences; they do not isolate a camera or prove memorization.

## Bottom-blanking sensitivity check

The completed stage-3 checkpoint was evaluated a second time with the bottom
20% of every test image blanked. This changes inference inputs only: the model
was **not trained** with the fingerprint-removal bundle.

| Evaluation | Disjoint BA | Shared BA | Gap [95% CI] |
|---|---:|---:|---:|
| Original images | 30.66% | 38.01% | +7.35 pp [5.55, 9.12] |
| Bottom 20% blanked | 20.9% | 23.9% | +3.0 pp [1.6, 4.6] |

Source for blanked outputs: `bottom_blanked/metrics.json` and
`bottom_blanked/predictions.csv`. Point estimates and intervals above are
rounded. Paired contributor bootstrap (2,000 draws, `scripts/compare_runs.py`)
gives the following changes for blanked minus original:

- Disjoint balanced accuracy: **−9.8 pp** [−10.8, −8.8].
- Shared-minus-disjoint gap: **−4.3 pp** [−5.8, −2.9].

The reduction in performance and gap is consistent with sensitivity to
information in the bottom image band, but blanking also removes useful scene
content and introduces an input shift. It cannot establish that the original
gap was caused by camera fingerprints, or substitute for training with and
without fingerprint removal.

## Reproduce this handoff

Requires local untracked dataset and checkpoint files:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -I scripts/final_results.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -I scripts/final_exposure.py
```

`scripts/final_results.py` verifies identical evaluated image IDs, countries and
contributor IDs across models before computing paired model comparisons.
