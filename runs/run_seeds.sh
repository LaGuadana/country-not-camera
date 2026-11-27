#!/bin/sh
# Seed repeats of the end-to-end pair (INSTRUCTIONS_FOR_CLAUDE_ON_MAC.md, task 1): one uninterrupted
# 8-epoch cosine schedule, lr 0.01/256 for all layers (head x1), 1 warm-up epoch, bf16 on MPS, one run at a time.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -I"; W="$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
for seed in 1 2; do
  for arm in "" _fp; do
    name=ft_imagenet${arm}_s$seed
    [ -f runs/$name/eval/metrics.json ] && continue
    FP=; [ "$arm" = _fp ] && FP=--fingerprint
    echo "$(date '+%T') $name training"
    $PY scripts/train.py --data-dir data/c32 --out-dir runs/$name --weights "$W" --init imagenet $FP \
        --epochs 8 --lr 0.01 --head-lr-mult 1 --warmup-epochs 1 --seed $seed --workers 4 --log-every 50 \
        --resume > logs/$name.log 2>&1 || { echo "$name FAILED"; exit 1; }
    $PY scripts/evaluate.py --ckpt runs/$name/best.pt --features --workers 4 > logs/eval_$name.log 2>&1 \
        || { echo "$name eval FAILED"; exit 1; }
    echo "$(date '+%T') $name done: $(grep '^gap test_shared' logs/eval_$name.log)"
  done
done
echo "$(date '+%T') ALL SEEDS DONE"
