# Learning the Country, Not the Camera: ResNet-18 on OSV-5M

Public repository: <https://github.com/LaGuadana/country-not-camera> (code MIT; data and
embedded photos CC-BY-SA-4.0, cite Astruc et al., CVPR 2024). Progress report:
[`report/progress/main.pdf`](report/progress/main.pdf); the playable demo is
[`game/dashcam_duel.html`](game/dashcam_duel.html) (see section 4).

Country classification from OpenStreetView-5M street-level images with a ResNet-18, either
fine-tuned from ImageNet or trained from scratch, evaluated on contributors the model never saw.
The earlier EDA and linear-probe baselines are in `~/ML/geo_cnn`. This folder holds the CNN
pipeline: subset selection, image download, the fingerprint-removal pipeline, training and
evaluation.

## Layout

```
geocnn/            library
  remote_zip.py      range-read single JPEGs from the OSV-5M zips on Hugging Face
  images.py          4:3 centre crop + resize to 128x170, packed storage (one .bin + index)
  fingerprint.py     static-pixel masks, bottom-band blanking
  data.py            Dataset (CPU: decode, blank, crop, JPEG) + Prep (GPU: colour jitter, normalise)
  model.py           CountryNet = torchvision ResNet-18 + dropout + linear head; param groups
  engine.py          inference loop
  metrics.py         top-1/top-3/balanced accuracy, contributor-clustered bootstrap
scripts/
  select_subset.py   manifest: 32 countries, contributor-disjoint splits, per-country budgets
  extend_train.py    all remaining capped training images, dealt into chunks (train_extra.csv)
  index_zips.py      offsets of every image in the 98 remote train zips
  fetch_images.py    download + downsample + pack the manifest's images (resumable)
  build_masks.py     per-contributor static-pixel masks
  train.py           fine-tune / scratch training, checkpoints, per-epoch val metrics
  linear_probe.py    frozen ImageNet features + logistic regression, saved as a checkpoint
  export_game.py     build the Dashcam Duel phone game from any checkpoint
  evaluate.py        test metrics with CIs, shared-minus-disjoint gap, feature export
  compare_runs.py    paired bootstrap of accuracy and gap changes between evaluated runs
kaggle/
  make_bundles.sh    stage the code and data as Kaggle datasets
  train_kaggle.ipynb GPU training notebook (one condition per notebook version)
game/template.html Dashcam Duel page; export_game.py fills in photos and guesses
data/c32/          the subset (not in git): manifest.csv, images.bin, masks.npz, stats
```

## Setup

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
PY=".venv/bin/python -I"     # -I: scripts read downloaded data
```

## 1. Data (local, ~1 h)

```sh
# manifest from the already-downloaded train.csv (2 min, peak RAM 2.4 GB)
$PY scripts/select_subset.py --train-csv ~/ML/geo_cnn/data/raw/train.csv --out-dir data/c32
# central directories of the 98 remote train zips (~450 MB of range reads, ~5 min)
$PY scripts/index_zips.py --out-dir data/zip_index
# 139,696 images, one range request each: 42 min at ~55 img/s from here, 1.0 GB on disk; resumable
$PY scripts/fetch_images.py --manifest data/c32/manifest.csv --index-dir data/zip_index --out data/c32/images.bin
$PY scripts/build_masks.py --data-dir data/c32      # 4 min; only used with --static-mask
```

What the subset is (defaults of `select_subset.py`, numbers from `data/c32/subset_stats.json`):

| split | images | contributors | per country | role |
|---|---|---|---|---|
| train | 101,578 | 11,132 | 3,200 (ET 2,649, MY 2,929) | training |
| val | 6,400 | 1,521 | 200 | model selection; contributors unseen |
| test | 12,800 | 3,043 | 400 | **contributor-disjoint** test |
| val_shared | 6,359 | 2,089 | ≤200 | unseen trips of training contributors |
| test_shared | 12,559 | 3,037 | ≤400 | **contributor-shared** test |

- Countries: the 32 with ≥200 contributors in the OSV-5M train split. All images come from
  the OSV-5M *train* split, because 98.2 % of the official test images were taken by
  contributors who also appear in train.
- Contributors are assigned 70/10/20 to train/val/test, stratified by their main country. No
  contributor appears in two of these splits, and no sequence is shared between train and any
  evaluation split. The script checks both and aborts on a violation.
- `*_shared` sets are whole groups of a training contributor's images (linked by sequence or by
  the same ~11 km cell) held out from training, so they test new trips and places by seen
  photographers. The gap `test_shared − test` measures the contributor shortcut.
- Per-country budgets are filled by water-filling: every contributor gets at most *c* images,
  with *c* as large as the budget allows. In the US that means ~1–2 images per contributor
  (1,719 contributors); in Malaysia up to ~50. The minimum effective number of contributors per
  country in train is 45.
- Storage: centre-crop to 4:3 (removes the 4:3 vs 16:9 camera cue; 16:9 frames lose 12.5 % per
  side), resize to 128×170, JPEG q90, ~7 KB per image.

## 2. Smoke test (CPU, ~3 min)

```sh
$PY scripts/fetch_images.py --manifest data/c32/manifest.csv --index-dir data/zip_index \
    --out data/smoke/images.bin --limit-per-split 300
ln -s ../c32/manifest.csv data/smoke/; ln -s ../c32/classes.json data/smoke/
$PY scripts/train.py --data-dir data/smoke --out-dir runs/smoke --init imagenet \
    --weights ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth --fingerprint \
    --epochs 2 --batch-size 32 --workers 2 --max-steps 5 --limit-eval 96
$PY scripts/evaluate.py --ckpt runs/smoke/best.pt --features --workers 2
```

## 3. Training on Kaggle

```sh
KAGGLE_USERNAME=<you> sh kaggle/make_bundles.sh   # stages dist/geocnn-code and data/c32
kaggle datasets create -p dist/geocnn-code         # private by default
kaggle datasets create -p data/c32                 # ~1 GB upload
```

Then open `kaggle/train_kaggle.ipynb` on Kaggle, attach both datasets, choose a GPU, set
`RUN` to one of `ft_imagenet`, `ft_imagenet_fp`, `scratch`, `scratch_fp`, and use
"Save & Run All". Each condition is one notebook version. A T4 x2 machine can run two
conditions at once (`CUDA_VISIBLE_DEVICES=0` / `1`) if the CPU keeps up.

Training details (`scripts/train.py`; these are starting points, not tuned):

| | ImageNet fine-tune | scratch |
|---|---|---|
| init | torchvision `IMAGENET1K_V1` (same weights as the linear probe) | random, zero-init residual BN |
| epochs | 30 | 60 |
| optimiser | SGD Nesterov 0.9, lr 0.02 per 256 images, head lr ×10 | SGD, lr 0.2 per 256, head ×1 |
| weight decay | 1e-4 (none on BN/bias) | 5e-4 |
| schedule | 1 epoch linear warm-up, cosine to 0 | 3 epochs warm-up, cosine |
| loss | cross-entropy, label smoothing 0.1, dropout 0.2 before the head | same |

- Input: 128×170, standard ResNet-18 unchanged (last feature map 4×6).
- Augmentation in both conditions: random resized crop (scale 0.6–1). **No horizontal flip**,
  because mirroring swaps the driving side and mirrors text.
- `--fingerprint` adds: bottom 20 % blanked to the ImageNet mean colour, JPEG re-encoding at
  quality 30–95 (p = 0.8), and per-image white-balance (±15 %), brightness/contrast/saturation
  (±25 %) jitter. `--static-mask` also blanks each contributor's static-pixel mask (opt-in; see
  Findings). Hue is not jittered because the colour of road
  markings is a country cue. The blanking (not cropping) keeps the image geometry identical
  across conditions, and it also applies at evaluation.
- Mixed precision (fp16 on T4/P100, bf16 on Ampere), channels-last.
- `best.pt` is chosen by balanced accuracy on `val`, the contributor-disjoint split.
- `--freeze-epochs N` trains only the head for N epochs first (linear probe then fine-tune).
- `--time-limit-min` + `--resume` continue across sessions: the data order and RNG state resume
  where they stopped.
- `metrics.csv` logs `val` and `val_shared` every epoch, plus their difference
  `shared_minus_disjoint_bacc`.

`evaluate.py` writes top-1/top-3/balanced accuracy with 95 % CIs from a cluster bootstrap over
contributors (Field & Welsh 2007), per-class recall, confusion matrices, the
`test_shared − test` gap with its CI, per-image predictions, and with `--features` the 512-d pooled
features plus contributor ids. The contributor probe trains on those features.

## Findings while building this

- **Per-contributor static-pixel masks barely find the camera.** On the local test shard, no
  pixel had an across-photo std below 15 for any of the 60 largest contributors, even those with
  a visible bonnet. Bonnets reflect the sky, and contributors change cars and mounts. On the
  subset, 3,674 of 15,696 contributors have enough images (≥8 from ≥4 sequences) for a mask,
  and only 226 masks were non-empty. The largest ones (18–34 % of the frame) covered road
  surface, snow or uniform sky, i.e. country cues, and their contributor-specific shapes would
  themselves be a fingerprint. Masks are now capped at 10 % of the frame (209 contributors,
  mean area 2.2 %, 2.5 % of images) and off by default. The bottom-band blanking handles
  bonnets and dashboards.
- The Hugging Face CDN serves ~55 range requests/s to this machine. Fetching whole shards
  instead would mean ~245 GB at the measured ~2.5 MB/s.
- pandas 3 `Series.isin` on string columns is ~30× slower than on object columns, hence
  `.astype(object)` before `isin` in a few places.

## 4. Linear probe and the phone game

```sh
# frozen ImageNet ResNet-18 features + logistic regression, 300 images per country (~7 min on CPU)
$PY scripts/linear_probe.py --data-dir data/c32 --out-dir runs/probe \
    --weights ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
# Dashcam Duel: 640 test photos (20 per country, one per contributor) dealt into 64 games of 10;
# the page (~90 KB) holds the model's top-3 guesses; game/g/<k>.json (~0.9 MB) holds each game's
# originals (512 px high, re-fetched once into data/c32/game_originals.bin), the 128x170 model
# input and a Grad-CAM map of the model's top guess (geocnn/explain.py). ~6 min on CPU.
$PY scripts/export_game.py --ckpt runs/probe/best.pt --out game/dashcam_duel.html \
    --model-label "ResNet-18 with ImageNet features and a linear probe"
```

Publish the page together with `game/g/*.json`. The page fetches a game's file when that game
starts and prefetches the next one. It remembers which games the phone has played, so photos
repeat only after all 64 games. The selection depends only on the data and `--seed`, so a
re-export with a new checkpoint keeps the same games and only the page changes.

Probe at 128×170: 16.7 % balanced accuracy on `val` (chance 3.1 %), top-3 33.7 %. The chosen L2
penalty (0.01) is the largest in the grid, so a wider grid may do slightly better. After the
Kaggle fine-tune, run `export_game.py` with that `best.pt` (and its `eval/` folder next to it) to
give the game the stronger opponent.

## 5. CPU fine-tune (interim, while Kaggle is not set up)

`runs/cpu_ft.sh` continues from the probe and trains layer3, layer4 and the head on this laptop
(stem and layer1–2 frozen: 17 vs 10 img/s for the full network on the 4-core i5). Each epoch is
20,480 random training images (~23 min); 24 epochs ≈ 4.7 passes over the set, ≈ 9 h. Progress is
in `logs/cpu_ft.log` and `runs/cpu_ft/metrics.csv` (val metrics on a fixed 1,600-image sample per
split). Re-run the script to resume after a reboot or sleep. This is not the end-to-end
fine-tune of the proposal; that one runs on Kaggle.

Result (run of 2026-10-12, 7.7 h; best epoch 18 of 24 by val balanced accuracy on the 1,600-image
sample; `evaluate.py` on the full splits, 95 % contributor-clustered bootstrap CIs):

| balanced acc. | test (unseen contributors) | test_shared (seen contributors) | gap shared − disjoint | top-3 test |
|---|---|---|---|---|
| linear probe (frozen ImageNet) | 13.5 % [12.8, 14.3] | 13.7 % [12.7, 14.7] | +0.2 pp [−1.1, 1.4] | 29.4 % |
| CPU fine-tune (layer3–4 + head) | 28.7 % [27.5, 29.8] | 34.5 % [33.1, 35.7] | **+5.8 pp [4.1, 7.5]** | 51.1 % |

- Fine-tuning doubles accuracy on unseen contributors. It also opens a significant gap in favour of
  contributors seen in training, which the frozen features do not show. The val_shared − val gap
  in `metrics.csv` was ~0 for the first 8 epochs and grew to 3–4 pp as train accuracy rose to
  54 %. The fingerprint-removal runs (`--fingerprint`) test whether that gap is camera cues.
- Best countries on test: Ethiopia 59 %, India 56 %, Taiwan 52 %, Japan 51 %. Worst: France 5 %,
  Czechia 12 %, Poland and Germany 13 %, Portugal 14 %. Their errors stay within Europe: Spain and
  Portugal go to Italy, Czechia to Poland and Hungary, Germany to the Netherlands and Denmark, and
  France is spread over Italy, Belgium, the UK and Denmark (`eval/confusion_test.csv`).

## 6. More training data under the same contributor rules

```sh
$PY scripts/extend_train.py --data-dir data/c32   # 1.5 min, ~2 GB RAM -> data/c32/train_extra.csv
runs/stages.sh                                    # fetch + train chunk by chunk; re-run to resume
```

`extend_train.py` recomputes the contributor split and held-out groups exactly as
`select_subset.py` did, and checks them against all 139,696 manifest rows. It then lists every
other image of a train contributor outside the held-out groups, keeping at most 100 per contributor
and country, existing images included. Evaluation splits are unchanged.

- Under that cap only **261,572** more training images exist, although the 32 countries have 3.8 M
  images in OSV-5M train. Most of their images come from a few heavy contributors. The extra images
  are uneven: US 60,637, DE 38,806, FR 23,368, IT 15,227, RU 15,044, … BE 1,019. ET and MY have none.
- They are dealt into 3 chunks of ~87k with the same mix of countries.
- `runs/stages.sh` trains one pass per chunk (`--chunk K`, 5 epochs whose `CoverageSampler` walks
  one permutation, so every image is seen once), starting from the previous stage's best checkpoint.
  It uses warm-up and cosine to 0, inverse-frequency class weights, and the same frozen
  stem/layer1–2. Stage 1 restarted at lr 0.01 per 256 and lost 3 pp of val balanced accuracy in
  its first epoch before recovering (a warm-restart dip). Stages 2–3 restart at 0.004 and showed no
  dip.
- Countries with fewer than 1,600 new images in a chunk are topped up with random earlier images
  (`--replay-floor`), so ET and MY are not forgotten. That gives 87,190 new + 15,932 earlier images
  per stage, ~1.8 h each on this CPU.
- The next chunk downloads while a stage trains (~22 img/s on 2026-10-12, ~65 min per chunk).

Results (full evaluation splits, balanced accuracy, 95% contributor-bootstrap CIs; chance 3.1%):

| run | train images | test | test_shared | gap shared − disjoint |
|---|---|---|---|---|
| CPU fine-tune (§5) | 101,578 | 28.7% [27.5, 29.8] | 34.5% | +5.8 pp [4.1, 7.5] |
| stage 1 | +87,190 | 29.8% [28.6, 31.0] | 36.8% | +7.0 pp [5.2, 8.8] |
| stage 2 | +87,194 | 30.1% [28.9, 31.3] | 37.7% | +7.6 pp [5.8, 9.4] |
| stage 3 (final) | +87,188 = 363,150 | **30.7%** [29.4, 31.9] | 38.0% | **+7.4 pp** [5.6, 9.2] |

`scripts/compare_runs.py runs/cpu_ft runs/stage_{1,2,3}` scores each pair on the same bootstrap
draws. Stage 3 against the CPU fine-tune: test **+2.0 pp [1.4, 2.5]**, gap **+1.6 pp [0.6, 2.5]**.

- 3.6× the images bought 2 pp on unseen contributors. Contributors the model had already seen gained
  3.5 pp, so the shortcut grew. The extra images are more trips of the same contributors (up to 100
  each), so they add contributor identity faster than new places. Most of the gap increase came in
  stage 1. That stage also switched on class weights, so it does not cleanly separate data from
  objective.
- Per-country test recall moved against the extra data (correlation with log extra images −0.41;
  400 test images per country, so single countries move ±2–4 pp by chance). ET +9.5 and MY +4.7 pp
  with no new images. IT −6.2, US −3.5 and DE −3.5 pp with the most. FR, the worst class, went from
  5% to 8%. Two likely reasons: class weights make each US image count ~1/12 of a replayed ET image,
  and replay shows ET/MY images again every stage.
- The game (`game/dashcam_duel.html`) now uses `runs/stage_3/best.pt`, which gets 31% of its 640
  photos right.

## 7. Anatomy of the shortcut (progress-report analyses)

Four scripts dissect *why* the shared−disjoint gap exists; all read the exported
`eval/features_*.npz` / `eval/predictions.csv` and run in minutes on CPU:

```sh
$PY scripts/contributor_probe.py runs/probe runs/cpu_ft runs/stage_{1,2,3}
$PY scripts/shortcut_decomposition.py runs/probe runs/cpu_ft runs/stage_3
$PY scripts/knn_probe.py runs/cpu_ft runs/stage_{1,2,3}
$PY scripts/error_geography.py runs/probe runs/cpu_ft runs/stage_{1,2,3}
$PY scripts/report_figures.py        # report/progress/figures/*.pdf
```

(the probe run's features were exported later: `evaluate.py --ckpt runs/probe/best.pt
--features`; see `logs/eval_probe_feats.log`.)

- **Exposure dose–response** (`shortcut_decomposition.py` → `runs/shortcut_bins.json`):
  bin each test_shared contributor by training images (manifest train + train_extra) and
  compare top-1 on their held-out trips *against the same countries' disjoint-test accuracy*
  (country-mix adjusted, contributor-clustered bootstrap CIs). Frozen probe: flat
  (+0.3 pp [−0.7, +1.4] at ≥52 images). Fine-tune: +6.7 [5.2, 8.1]; stage 3: **+8.8 pp
  [+7.3, +10.3]** at ≥52 training images (median exposure 99; 84% of test_shared images),
  and −3.7 pp [−7.0, −0.3] at 7–21 images. The shortcut is memorisation of heavily-seen
  contributors, not geography.
- **Retrieval view** (`knn_probe.py` → `runs/knn_probe.json`): 1-NN/15-NN country
  classification on z-scored cosine features, gallery val→test (all contributors disjoint)
  vs. val_shared→test_shared. Stage-3 features: 15-NN 25.3% top-1 on test vs 30.7% for the
  head; on test_shared 6.9% of queries retrieve a *same-contributor* photo first — those
  queries are **90.6%** correct, the rest **22.0%** — contributing 0.069 × (0.906−0.220) ≈
  **+4.7 pp** of the +7.0 pp top-1 gap. The same pattern exists on frozen ImageNet features
  (5.0% of queries, then 89.6% vs 7.0% correct), yet the probe head shows no gap: the
  retrieval cue is available to any photographer-clustered representation, but only a
  classifier fitted to the contributors converts it into an accuracy gap.
- **Feature-space photographer verification** (`contributor_probe.py` → per-run
  `eval/contributor_probe.json`): cosine similarity of same-photographer pairs (different
  sequences) vs same-country different-photographer pairs; AUC 0.5 = chance. Frozen ImageNet
  features score **0.59/0.62** (test/test_shared), *higher* than fine-tuned (0.56–0.57), and
  NN-same-contributor rates are 11–17% vs 0.1% random-pair chance — yet the probe shows no
  gap and no exposure trend. Photographer-decodable features alone do not create the
  shortcut; fitting the classifier to specific contributors does.
- **Error geography** (`error_geography.py` → `runs/error_geography.json`): distance from
  each misclassified test image to the predicted country's centroid (mean of unit vectors).
  Median error distance halves from 5,543 km (probe) to 2,439 km (fine-tune), then drifts to
  2,865 km with more data; correct answers sit ~220 km from their centroid; share of errors
  beyond 4,000 km falls 0.55 → 0.43.

The progress report itself is `report/progress/main.tex` (+ `references.bib`,
`figures/`); compile with `tectonic main.tex` → 2 pages + references.
