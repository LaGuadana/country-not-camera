#!/bin/sh
# CPU fine-tune started 2026-10-12: from the linear probe (runs/probe), stem + layer1-2 frozen,
# layer3, layer4 and head trained. Re-run this script to resume after a reboot or sleep.
cd "$(dirname "$0")/.." || exit 1
OMP_NUM_THREADS=4 exec .venv/bin/python -I scripts/train.py --data-dir data/c32 --out-dir runs/cpu_ft \
  --init imagenet --weights "$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth" \
  --init-from runs/probe/best.pt --freeze-stages 2 \
  --epoch-size 20480 --epochs 24 --batch-size 64 --lr 0.02 --head-lr-mult 1 --warmup-epochs 0.5 \
  --workers 2 --limit-eval 1600 --log-every 80 --resume
