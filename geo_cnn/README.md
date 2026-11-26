# geo_cnn: country-from-street-photo, data + EDA + first baselines

Dataset: OpenStreetView-5M (OSV-5M), https://huggingface.co/datasets/osv5m/osv5m (CC-BY-SA-4.0).
All commands run from this folder (`/home/betzler/ML/geo_cnn`). Scripts take every path as an
argument; Python that reads downloaded files is run with `python -I`.

## 1. Environment

```sh
python3 -m venv .venv
.venv/bin/pip install pandas numpy scikit-learn matplotlib pillow requests huggingface_hub pycountry
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=".venv/bin/python -I"
```

## 2. Download (about 3.5 GB; resumable)

```sh
mkdir -p data/raw/images_test data/processed figures results
B=https://huggingface.co/datasets/osv5m/osv5m/resolve/main
curl -sSL -o data/raw/README_hf.md $B/README.md
src/fetch.sh $B/test.csv                data/raw/test.csv               116469997
src/fetch.sh $B/images/test/04.zip      data/raw/images_test/04.zip     453288134
src/fetch.sh $B/train.csv               data/raw/train.csv              2915296639
sha256sum -c data/SHA256SUMS.txt   # hashes = HF LFS oids
```

`images/test/04.zip` is the smallest image shard (453 MB). The zip is never extracted; scripts read
members directly from it.

## 3. EDA

```sh
# (a) country counts + long tail, (b) world scatter of 50k train coordinates
$PY src/eda_metadata.py --train data/raw/train.csv --test data/raw/test.csv \
    --results results --figdir figures --processed data/processed
# index the shard (id, size, metadata join) + (d) resolution figure
$PY src/index_shard.py --zip data/raw/images_test/04.zip --meta data/raw/test.csv \
    --out data/processed/shard_index.csv --figdir figures --results results
# (c) example grid
$PY src/fig_examples.py --zip data/raw/images_test/04.zip --index data/processed/shard_index.csv \
    --figdir figures --countries JP,BR,RU,IN,ZA,NO --per-country 2
# redraw (a)/(b) from saved results without rescanning the 2.9 GB CSV:
$PY src/eda_metadata.py --train data/raw/train.csv --test data/raw/test.csv \
    --results results --figdir figures --processed data/processed --plot-only
```

## 4. Balanced 10-country subset, split, features, baselines

```sh
$PY src/make_split.py --index data/processed/shard_index.csv --out data/processed/split.csv --results results
$PY src/features.py --kind hsv      --zip data/raw/images_test/04.zip --split data/processed/split.csv --out data/processed/feat_hsv.npz
$PY src/features.py --kind resnet18 --zip data/raw/images_test/04.zip --split data/processed/split.csv --out data/processed/feat_resnet18.npz --threads 2 > data/resnet.log 2>&1
$PY src/baselines.py --split data/processed/split.csv \
    --feat hsv=data/processed/feat_hsv.npz --feat resnet18=data/processed/feat_resnet18.npz \
    --results results --figdir figures

# STRICT sensitivity check: also keep photographers (creator_id) on one side.
# Same 1,620 images, so the feature files are reused.
$PY src/make_split.py --index data/processed/shard_index.csv --out data/processed/split_strict.csv \
    --results results --group-creator --tag _strict
$PY src/baselines.py --split data/processed/split_strict.csv \
    --feat hsv=data/processed/feat_hsv.npz --feat resnet18=data/processed/feat_resnet18.npz \
    --results results --figdir figures --tag _strict \
    --split-label "split: sequence-, 0.1° cell- and photographer-disjoint"

# collect all numbers (with sources) into numbers.txt; it also parses data/resnet.log if
# the ResNet step was run as: ... features.py --kind resnet18 ... > data/resnet.log 2>&1
$PY src/make_numbers.py --results results --processed data/processed --out numbers.txt
```

Run time on this laptop (CPU shared with another job, 2 threads): metadata EDA about 80 s,
shard index about 30 s, HSV features about 90 s, ResNet-18 features about 5 min, baselines < 1 min.
Peak RSS stayed below 0.75 GB (ResNet step).

Outputs: `figures/*.pdf|png`, `results/*.json|csv`, `numbers.txt` (generated), `NOTES.md` (facts, hand-maintained).

## Layout

- `src/` scripts (`style.py` = shared figure style, `countries.py` = ISO code to name)
- `data/raw/` untouched downloads (untrusted data), `data/processed/` derived tables/features
- `results/` JSON/CSV with every computed number, `figures/` PDF + PNG
