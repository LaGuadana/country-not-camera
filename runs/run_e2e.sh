#!/bin/sh
# End-to-end ResNet-18 fine-tune on Apple-silicon MPS (bf16 autocast), standard vs fingerprint removal.
#
# Phase 1 (this script, as run on 2026-11-25): both runs concurrently with an 8-epoch cosine schedule,
# lr 0.01 per 256 images for all layers (the train.py default, 0.02/256 with head lr x10, collapsed to
# chance accuracy in a trial). Concurrent runs made the 16 GB machine swap (13-21 min per epoch), so both
# were stopped by hand after epoch 4 and finished one after the other by runs/finish_e2e.sh, which resumes
# from last.pt with the schedule shortened to 6 epochs (identically for both runs).
# The planned from-scratch pair (--init scratch) was not run.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -I"; W="$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
run() { name=$1; shift
  $PY scripts/train.py --data-dir data/c32 --out-dir runs/$name --weights "$W" --init imagenet \
      --epochs 8 --lr 0.01 --head-lr-mult 1 --workers 3 --log-every 50 --resume "$@" > logs/$name.log 2>&1
  echo "$(date '+%T') $name stopped"; }
run ft_imagenet &
sleep 15
run ft_imagenet_fp --fingerprint &
wait
