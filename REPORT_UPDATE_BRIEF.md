# Brief: results to write into the final report (27 Nov 2026)

All runs below finished on the Apple M5 (MPS, bf16 training, fp32 evaluation) and are evaluated on the
same 38,118 evaluation images as every earlier model. Balanced accuracy (BA) in %, 95% photographer-
clustered bootstrap CIs; "paired" = same bootstrap draws. Follows `INSTRUCTIONS_FOR_CLAUDE_ON_MAC.md`.

## 1. Seed repeats of the end-to-end pair (clean 8-epoch schedule, no restart)
Config: ImageNet init, all layers, SGD Nesterov 0.9, batch 256, lr 0.01/256 (head x1), 1 warm-up epoch,
cosine, wd 1e-4, dropout 0.2, label smoothing 0.1, crops 0.6-1, no flips; `runs/run_seeds.sh`.

| run | test BA | test_shared BA | gap | paired FP effect on gap | paired FP effect on test |
|---|---|---|---|---|---|
| original pair (6 ep, restarted, `runs/ft_imagenet*`) | 31.9 -> 28.2 | 37.0 -> 31.3 | +5.2 -> +3.1 | -2.1 [-3.3, -0.9] | -3.6 [-4.5, -2.8] |
| seed 1 (`ft_imagenet_s1`, `ft_imagenet_fp_s1`) | 31.8 -> 29.0 | 38.5 -> 31.7 | +6.7 -> +2.7 | **-4.0 [-5.2, -2.7]** | -2.8 [-3.6, -1.9] |
| seed 2 (`ft_imagenet_s2`, `ft_imagenet_fp_s2`) | 32.1 -> 28.8 | 37.7 -> 32.6 | +5.6 -> +3.8 | **-1.8 [-3.0, -0.6]** | -3.3 [-4.2, -2.5] |

The effect replicates in all three runs (CIs exclude 0); size varies 1.8-4.0 pp by seed. Report each
run and the range; keep the original pair labelled as the first run with a disrupted schedule.
Files: `results/final/paired_fp_vs_standard_s1.json`, `..._s2.json`, runs' `eval/metrics.json`.

## 2. Fingerprint ablation (seed 1, same schedule as seed-1 pair)
`--fp-parts blank` and `--fp-parts jpeg,color` (new option in `scripts/train.py`).

| arm | test BA | gap | paired vs standard: test | paired vs standard: gap |
|---|---|---|---|---|
| standard | 31.8 | +6.7 | | |
| blanking only (`ft_imagenet_fpblank_s1`) | 29.3 | +4.2 | -2.4 [-3.2, -1.6] | -2.6 [-3.8, -1.5] |
| JPEG + colour only (`ft_imagenet_fpjitter_s1`) | 31.2 | +5.1 | -0.5 [-1.2, 0.2] | -1.6 [-2.7, -0.5] |
| full (`ft_imagenet_fp_s1`) | 29.0 | +2.7 | -2.8 [-3.6, -1.9] | -4.0 [-5.2, -2.7] |

Both parts reduce the gap and roughly add up; JPEG + colour costs almost no accuracy, blanking costs
~2.4 pp (the bottom band shows road as well as bonnet). File: `results/final/paired_ablation_s1.json`.

## 3. From-scratch pair (completes the {ImageNet, scratch} x {standard, FP} design)
60 epochs, lr 0.2/256, wd 5e-4, 3 warm-up epochs (train.py scratch defaults), seed 1.

| run | test BA | test_shared BA | gap |
|---|---|---|---|
| scratch (`scratch_s1`) | 28.8 | 38.8 | **+10.0 [8.3, 11.9]** |
| scratch + FP (`scratch_fp_s1`) | 25.3 | 30.8 | +5.5 [3.9, 7.2] |

Paired: FP changes scratch test by -3.5 [-4.4, -2.6] and gap by **-4.5 [-5.8, -3.2]**. Scratch vs
ImageNet (seed 1): test -2.9 [-3.8, -2.0], gap **+3.3 [1.9, 4.6]**: without pretrained features the
network leans more on the photographer (proposal's prediction). Scratch-FP trains to 98% train top-1
(heavy overfitting). Files: `paired_scratch_fp_vs_standard.json`, `paired_scratch_vs_imagenet_s1.json`.

## 4. Test-time band controls (blank bottom 20% vs top 20%, no retraining)
| model | bottom blanked: test / gap change | top blanked: test / gap change |
|---|---|---|
| ft_imagenet (original) | -9.2 / **-2.4 [-3.8, -1.1]** | -5.5 / +0.4 [-0.8, 1.5] |
| ft_imagenet_s1 | -9.4 / **-4.2 [-5.5, -2.8]** | -5.3 / -1.1 [-2.2, 0.1] |
| ft_imagenet_s2 | -8.0 / **-3.4 [-4.6, -2.1]** | -7.9 / +0.7 [-0.5, 1.9] |

Removing an equal-sized area elsewhere costs accuracy but does not reduce the gap: the gap reduction is
specific to the bonnet band, not a generic effect of missing pixels. Files: `paired_bands_<run>.json`.

## 5. Other completed items
- Probe refit, wider grid (`runs/probe_wide`): optimum L2 0.03 (interior); test 14.1 [13.3, 14.9],
  test_shared 13.5, gap -0.6 [-1.8, 0.8], top-3 30.0. No gap, as before. Trained on a 300/class sample of
  the re-drawn split. Remove the "optimum at the edge of the grid" caveat; update row (2) or add a row.
- Reproduction: stage 3 re-evaluated reproduces 38,117 of 38,118 committed predictions; the one change
  is an India/Russia near-tie (logit margin 0.09) on a Mexican image, wrong either way, metrics identical.
  Replace "all 38,118 exactly".
- Distance x exposure (`results/final/distance_exposure.json`, `..._new.json`): at <25 km the excess
  rises with exposure from -0.3 (2-5 training images) to +11.1 pp (26+); for 26+ it falls with distance
  (+11.1, +6.8, +3.7 at >=100 km). Advantage needs a familiar photographer AND nearby roads. Multi-country
  photographers are 11% of the least-exposed bin; excluding them moves -3.7 to -2.1.
  Place-vs-camera for new runs: `results/final/place_vs_camera_new.json` (scratch: +13.7 pp at <25 km).
- Per-country evidence (`results/final/country_features.json`, `scripts/country_features.py`): bottom
  fifth carries 10% (HU) to 30% (DK) of true-class Grad-CAM evidence; top quarter most for HU, RU, GR.
  Per-country evidence location does not predict the per-country gap or FP effect (Spearman |r| <= 0.23,
  p >= 0.2). Figures: `report/final/figures/fig_country_cam`, `fig_pairs` (AT vs CH, CA vs US evidence
  maps), `fig_confusion` (62% of errors within region block).

## 6. Report sections to update (`report/final/main.tex`, mirror in `report/final_bam/main.tex`)
- Abstract, conclusion: replicated FP effect (3 runs), scratch gap +10.0, full 2x2.
- 3.2 Models / Table 2: add seed runs (8 epochs), ablation, scratch column; 1.4 and Limitations: the
  from-scratch arm and probe refit are now done; keep "one split".
- 4.2 Table 4: rows for seeds, scratch pair, probe refit.
- 4.3: add per-country evidence figure and confusable-pair figure; 4.4: confusion-matrix figure.
- 5.2 Place or camera: add distance x exposure table and the multi-country check.
- 6.1 At test time: top-band control. 6.2 During training: per-seed table, ablation, scratch pair.
- 7 Discussion, 9 Future work: update (2x2 done, seeds done; remaining: more seeds, higher resolution,
  adversarial removal). README: section on the new runs (`runs/run_seeds.sh`, `runs/run_queue.sh`).
