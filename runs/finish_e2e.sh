#!/bin/sh
# Finish the end-to-end pair sequentially (concurrent runs thrashed swap on a 16 GB machine):
# resume each from last.pt (4 epochs done) with the cosine schedule shortened to 6 epochs, then evaluate.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -I"; W="$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
for n in ft_imagenet ft_imagenet_fp; do
  FP=; [ $n = ft_imagenet_fp ] && FP=--fingerprint
  $PY scripts/train.py --data-dir data/c32 --out-dir runs/$n --weights "$W" --init imagenet $FP \
      --epochs 6 --lr 0.01 --head-lr-mult 1 --workers 4 --log-every 50 --resume >> logs/$n.log 2>&1
  echo "$(date '+%T') $n trained"
done
for n in ft_imagenet ft_imagenet_fp; do
  $PY scripts/evaluate.py --ckpt runs/$n/best.pt --features --workers 4 > logs/eval_$n.log 2>&1
done
$PY scripts/evaluate.py --ckpt runs/ft_imagenet/best.pt --blank-bottom 0.2 --workers 4 > logs/eval_blank_ft_imagenet.log 2>&1
echo "$(date '+%T') evaluated"
