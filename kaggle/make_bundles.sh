#!/bin/sh
# Stage the two private Kaggle datasets the training notebook reads:
#   dist/geocnn-code  code + ResNet-18 ImageNet weights (so the notebook needs no internet)
#   data/c32          the packed images, manifest and masks (adds dataset-metadata.json there)
# Uploading is a separate, manual step (Kaggle CLI + API token in ~/.kaggle/kaggle.json):
#   kaggle datasets create -p dist/geocnn-code            # later: kaggle datasets version -p dist/geocnn-code -m "update"
#   kaggle datasets create -p data/c32
# Datasets created this way are private. OSV-5M is CC BY-SA 4.0: keep attribution if you ever share it.
#
# Usage: KAGGLE_USERNAME=<you> sh kaggle/make_bundles.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
KUSER=${KAGGLE_USERNAME:?set KAGGLE_USERNAME to your Kaggle user name}
WEIGHTS=${WEIGHTS:-$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth}
DATA=${DATA_DIR:-$ROOT/data/c32}

CODE=$ROOT/dist/geocnn-code
rm -rf "$CODE"
mkdir -p "$CODE"
cp -r "$ROOT/geocnn" "$ROOT/scripts" "$ROOT/requirements.txt" "$CODE/"
find "$CODE" -name __pycache__ -prune -exec rm -rf {} +
cp "$WEIGHTS" "$CODE/resnet18-f37072fd.pth"
cat > "$CODE/dataset-metadata.json" <<EOF
{"title": "geocnn-code", "id": "$KUSER/geocnn-code", "licenses": [{"name": "other"}]}
EOF

for f in manifest.csv classes.json images.bin images_index.csv; do
  [ -f "$DATA/$f" ] || { echo "missing $DATA/$f (run the data pipeline first)"; exit 1; }
done
[ -f "$DATA/masks.npz" ] || echo "note: $DATA/masks.npz missing; needed only for --static-mask"
cat > "$DATA/dataset-metadata.json" <<EOF
{"title": "osv5m-c32-128", "id": "$KUSER/osv5m-c32-128", "licenses": [{"name": "CC-BY-SA-4.0"}],
 "description": "Contributor-capped, contributor-disjoint 32-country subset of OpenStreetView-5M (Astruc et al., CVPR 2024; https://huggingface.co/datasets/osv5m/osv5m; CC BY-SA 4.0; images from Mapillary), centre-cropped to 4:3 and resized to 128x170."}
EOF
du -sh "$CODE" "$DATA"
echo "staged. Upload with:"
echo "  kaggle datasets create -p $CODE"
echo "  kaggle datasets create -p $DATA"
