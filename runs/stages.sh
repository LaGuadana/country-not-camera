#!/bin/sh
# Continue the CPU fine-tune on the extra chunks of data/c32/train_extra.csv (scripts/extend_train.py):
# one pass per chunk, each stage starting from the previous stage's best checkpoint, while the next
# chunk downloads. Re-run to resume: fetching skips packed images, training resumes from last.pt,
# finished stages (runs/stage_K/done) are skipped.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -I"
N=$($PY -c "import pandas as pd; print(pd.read_csv('data/c32/train_extra.csv', usecols=['chunk'])['chunk'].max())")

fetch() {  # chunks 1..$1, up to 3 tries
  for try in 1 2 3; do
    $PY scripts/fetch_images.py --manifest data/c32/train_extra.csv --max-chunk "$1" \
      --index-dir data/zip_index --out data/c32/images.bin && return 0
  done
  return 1
}

# Stage 1 ran at --lr 0.01 and lost 3 pp of val balanced accuracy in its first epoch before
# recovering (a warm-restart dip); later stages restart at a lower peak.
lr() { [ "$1" -eq 1 ] && echo 0.01 || echo 0.004; }

prev=runs/cpu_ft/best.pt
for k in $(seq 1 "$N"); do
  out=runs/stage_$k
  if [ ! -f "$out/done" ]; then
    echo "$(date '+%F %T') stage $k: fetching chunk $k"
    fetch "$k" >> logs/fetch_extra.log 2>&1 || { echo "fetch of chunk $k failed"; exit 1; }
    echo "$(date '+%F %T') stage $k: training from $prev"
    OMP_NUM_THREADS=4 $PY scripts/train.py --data-dir data/c32 --out-dir "$out" \
      --init imagenet --weights "$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth" \
      --init-from "$prev" --chunk "$k" --replay-floor 1600 --class-weighted --freeze-stages 2 \
      --split-pass 5 --epochs 5 --batch-size 64 --lr "$(lr "$k")" --head-lr-mult 1 --warmup-epochs 0.25 \
      --workers 2 --limit-eval 1600 --log-every 80 --resume >> "logs/stage_$k.log" 2>&1 &
    train=$!
    if [ "$k" -lt "$N" ]; then
      sleep 180  # let the trainer load its index first
      fetch $((k + 1)) >> logs/fetch_extra.log 2>&1 || echo "prefetch of chunk $((k + 1)) failed; retrying later"
    fi
    wait $train || { echo "stage $k failed (see logs/stage_$k.log)"; exit 1; }
    touch "$out/done"
    echo "$(date '+%F %T') stage $k done: $(tail -1 "logs/stage_$k.log")"
  fi
  prev=$out/best.pt
done
echo "$(date '+%F %T') all $N stages done"
