#!/bin/sh
# GPU queue after the seed repeats (one job at a time): fingerprint ablation, test-time band controls,
# then the from-scratch pair that completes the {ImageNet, scratch} x {standard, FP removal} design.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -I"; W="$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
until grep -q "ALL SEEDS DONE" logs/run_seeds.log; do grep -q FAILED logs/run_seeds.log && exit 1; sleep 30; done
train() { name=$1; shift
  [ -f runs/$name/eval/metrics.json ] && return 0
  echo "$(date '+%T') $name training"
  $PY scripts/train.py --data-dir data/c32 --out-dir runs/$name --weights "$W" --workers 4 --log-every 50 \
      --resume "$@" > logs/$name.log 2>&1 || { echo "$name FAILED"; return 1; }
  $PY scripts/evaluate.py --ckpt runs/$name/best.pt --features --workers 4 > logs/eval_$name.log 2>&1 \
      || { echo "$name eval FAILED"; return 1; }
  echo "$(date '+%T') $name done: $(grep '^gap test_shared' logs/eval_$name.log)"; }
E2E="--init imagenet --epochs 8 --lr 0.01 --head-lr-mult 1 --warmup-epochs 1 --seed 1"
# 1. fingerprint ablation (seed 1, same schedule as ft_imagenet_s1 / ft_imagenet_fp_s1)
train ft_imagenet_fpblank_s1 $E2E --fingerprint --fp-parts blank
train ft_imagenet_fpjitter_s1 $E2E --fingerprint --fp-parts jpeg,color
# 2. test-time band controls
for r in ft_imagenet ft_imagenet_s1 ft_imagenet_s2 stage_3; do
  [ -f runs/$r/eval_blank/metrics.json ] || $PY scripts/evaluate.py --ckpt runs/$r/best.pt --blank-bottom 0.2 --workers 4 > logs/eval_blank_$r.log 2>&1
  [ -f runs/$r/eval_blank_top/metrics.json ] || $PY scripts/evaluate.py --ckpt runs/$r/best.pt --blank-top 0.2 --workers 4 > logs/eval_blank_top_$r.log 2>&1
done
echo "$(date '+%T') band controls done"
# 3. from-scratch pair (train.py defaults for scratch: 60 epochs, lr 0.2/256, wd 5e-4, 3 warm-up epochs)
train scratch_s1 --init scratch --seed 1
train scratch_fp_s1 --init scratch --fingerprint --seed 1
echo "$(date '+%T') QUEUE DONE"
