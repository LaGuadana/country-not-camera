# NOTES (facts only; exact values with sources are in numbers.txt)

## Dataset
- OpenStreetView-5M (OSV-5M); Astruc, Dufour, Siglidis, ... Landrieu; "OpenStreetView-5M: The Many Roads to Global Visual Geolocation", CVPR 2024.
- Hugging Face repo: https://huggingface.co/datasets/osv5m/osv5m (API listing: https://huggingface.co/api/datasets/osv5m/osv5m). Last modified 2024-04-27. Not gated.
- Project page: https://imagine.enpc.fr/~ioannis.siglidis/osv5m/ ; demo: https://huggingface.co/spaces/osv5m/plonk (both linked from the HF README).
- Licence: CC-BY-SA-4.0 (HF card YAML `license: cc-by-sa-4.0`; copy of the card in data/raw/README_hf.md). Images come from Mapillary. Share-alike + attribution apply to anything redistributed (e.g. example images in a report: cite dataset + licence).
- Repo files: train.csv (2.92 GB), test.csv (116 MB), images/train/00..97.zip (98 shards), images/test/00..04.zip (5 shards; 00-03 about 2.24 GB each, 04 = 453 MB). Repo total about 259 GB.
- Metadata columns (26): id, latitude, longitude, thumb_original_url, country, sequence, captured_at, lon_bin, lat_bin, cell, land_cover, road_index, drive_side, climate, soil, dist_sea, region, sub-region, city, unique_city, unique_sub-region, unique_region, unique_country, quadtree_10_1000, creator_username, creator_id.
- `country` is an ISO 3166-1 alpha-2 code. No "NA" (Namibia) code occurs in either CSV, and no train-sample/test image lies in a Namibia bounding box; scripts still read with keep_default_na=False so a literal "NA" code could never turn into a missing value.
- Image file name inside the zips = `<id>.jpg`.
- Train: 4,894,684 images, 1,391,425 sequences, 222 countries. Test: 210,122 images, 80,950 sequences, 219 countries. Union: 225 countries (matches the "225 countries" claim).
- 3 countries only in test (KY, TO, TV); 6 only in train (AD, GI, LI, MC, MO, SM: microstates and small territories).
- Train is very imbalanced: US alone = 24.9 % of train; top 10 = 58.3 %; 142 countries have >=1k images, 58 have >=10k, 11 have >=100k; 37 have <100.
- Test country mix differs strongly from train (total-variation distance 0.459): test top 10 = US, RU, AU, BR, CA, IN, MX, CN, AR, KZ (US 11.4 %, top 10 = 44.6 %), train top 10 = US, DE, FR, RU, JP, AU, BR, GB, SE, IT. Examples: DE 5.56 % of train vs 0.57 % of test; JP 4.27 % vs 1.01 %; CN 0.16 % vs 2.10 %; KZ 0.15 % vs 1.85 %.
- Train images available for the 10 countries used below: US 1,218,664; RU 222,351; AU 162,138; BR 140,599; CA 84,299; MX 53,468; IN 50,604; AR 41,581; CN 7,757; KZ 7,545.

## What was downloaded (data/raw/, all SHA-256 verified against HF LFS oids, see data/SHA256SUMS.txt)
- test.csv 116,469,997 B; train.csv 2,915,296,639 B; images/test/04.zip 453,288,134 B (smallest image shard); README_hf.md; loader script osv5m_loader.py.txt (saved as text, never executed). Total about 3.49 GB.
- ResNet-18 ImageNet weights (44.7 MB) went to ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth.
- Downloads from HF were cut off several times ("transfer closed"); src/fetch.sh resumes with curl -C -.
- The zip is not extracted; all scripts read JPEGs directly from it.

## The downloaded image shard (images/test/04.zip)
- 10,122 JPEGs, all matched to test.csv; 194 countries, 8,928 sequences, 1,906 photographers (creator_id).
- Country mix of the shard is representative of the full test split (total-variation distance 0.043; same top-10 countries in the same order).
- Resolution: height is 512 px for almost all images; 682x512 (4:3) = 48.2 %, 910x512 (16:9) = 40.3 %; 99.2 % landscape; mean JPEG 44 KB.

## Sample design for the first baseline
- Classes: the 10 countries with most images in the shard: US, RU, AU, BR, CA, IN, MX, CN, AR, KZ.
- Balanced by random downsampling (seed 0) to n = 162 per country (= count of the 10th country, KZ); 1,620 images total.
- Leakage control (MAIN split): images linked if same Mapillary `sequence` OR same 0.1 x 0.1 degree lat/lon cell (about 11 km); connected components = groups; StratifiedGroupKFold(5), one fold held out. Result: 1,296 train / 324 test (32-33 per class), 0 sequences and 0 groups shared between train and test.
- Remaining overlap in MAIN split: photographers. 109 photographers appear in both parts; 278 of 324 test images have a photographer who is also in train.
- STRICT split (sensitivity check): also link by photographer (creator_id). 1,271 train / 349 test, 0 shared photographers, but the test set is no longer balanced (AU 96, KZ 13, CA 18, others 30-32) because a few photographers contribute many images (largest group = 120 images). For this split, balanced accuracy (mean per-class recall) is the fairer number.
- Both train and test images of this baseline come from the OSV-5M *test* split (only a test shard was downloaded). This is a within-shard split, not the official OSV-5M train/test protocol.

## Models and results (top-1 / top-3 on held-out images; chance = 10 % / 30 %)
- Majority class (train mode): MAIN 9.9 % / 29.6 %.
- Multinomial logistic regression on HSV histograms (16 H x 4 S x 4 V = 256 bins on a <=256 px thumbnail, L1-normalised, square-rooted, standardised; C by grouped 3-fold CV on train): MAIN 37.3 % / 66.0 % (95 % CI top-1 32.3-42.7 %). STRICT 16.9 % / 43.0 %, balanced 19.3 %.
- Frozen ImageNet ResNet-18 (torchvision IMAGENET1K_V1; JPEG decoded with PIL draft mode at >=448 px, then resize short side 224 + centre crop 224, ImageNet normalisation; 512-d pooled features) + logistic regression: MAIN 48.1 % / 77.8 % (95 % CI top-1 42.8-53.6 %). STRICT 26.6 % / 61.3 %, balanced 28.1 %.
- Best C = 0.01 for both feature sets on MAIN (grid 0.001, 0.01, 0.1, 1).
- Per-class recall, ResNet-18, MAIN: AU 81 %, KZ 72 %, CA 59 %, IN 58 %, CN 47 %, RU 47 %, BR 36 %, MX 30 %, US 27 %, AR 25 %.
- Under STRICT, AU recall drops 81 % -> 19 % and KZ 72 % -> 15 %: the MAIN-split accuracy is inflated by photographer/camera identity (same dashcam, same car bonnet, same colour profile), not only by country cues.

## Compute (measured on this laptop while an RL training job used most of the CPU)
- ResNet-18 feature extraction, CPU, 2 threads, batch 32, 224 px: forward pass only 7.9 img/s; end-to-end (JPEG decode from zip + preprocessing + forward) about 6.6 img/s in steady state (5.3 img/s over the whole run including the one-off 45 MB weight download); peak RSS 0.72 GB; 1,620 images in about 5 min.
- HSV histograms: 18.4 img/s end-to-end (1 process).
- Metadata EDA over the 2.9 GB train.csv: about 80 s, peak RSS about 0.32 GB (4 columns, 500k-row chunks).
- Extrapolation (not measured): frozen ResNet-18 features for 100k images at about 6.6 img/s would take about 4.2 h on this machine under the same load.

## Figures (figures/, each as PDF + PNG)
- fig_a_country_counts: left, top-30 countries of the train split (horizontal bars, log x); right, rank vs count for all 222 train countries with 1k/10k thresholds.
- fig_b_world_scatter: lon/lat of 50,000 random train images (no basemap). Shows dense coverage of US, Europe, Japan, SE Australia, NZ, south-east Brazil; sparse Africa, central Asia, Amazon; China nearly empty (CN = 0.16 % of train images).
- fig_c_examples: 2 random test-shard images each for Japan, Brazil, Russia, India, South Africa, Norway (centre-cropped to 4:3). Visible dashcam artefacts: car bonnet, windscreen reflections, timestamp overlay (Japan).
- fig_d_resolution: most common resolutions and aspect-ratio histogram of the 10,122 shard images.
- fig_e_confusion_best: row-normalised confusion matrix, ResNet-18 + logistic regression, MAIN split.
- fig_e_confusion_best_strict: same model, STRICT (photographer-disjoint) split.

## Caveats
- Small test sets (324 / 349 images): 95 % CIs on top-1 are about +-5 percentage points.
- One random seed, one held-out fold; no repeated splits yet.
- Photographer leakage is large (see STRICT numbers); the proposal's evaluation protocol should group by photographer, or use the official OSV-5M train/test separation.
- Country label is derived from the GPS position in the metadata; label noise near borders not checked.
- Dashcam artefacts (bonnet, overlays, timestamps) are possible shortcuts for any model.
- Centre crop to 224 px discards the left/right sides of 4:3 and 16:9 images.
- Throughput numbers were measured with the CPU shared with another job; an idle machine would be faster.
- The test split's country distribution differs from train; a model trained on the official train split faces a prior shift on the official test split.
