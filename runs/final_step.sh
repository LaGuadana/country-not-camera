#!/bin/sh
# All final-report analyses once runs/ft_imagenet{,_fp} are evaluated (outputs: results/final/,
# report/final/figures/). Needs the eval features/logits, which evaluate.py writes but git does not track.
cd "$(dirname "$0")/.." || exit 1
A=".venv/bin/python -I scripts/final_analysis.py"
$A paired runs/ft_imagenet/eval runs/ft_imagenet_fp/eval --name fp_vs_standard
$A paired runs/ft_imagenet/eval runs/ft_imagenet/eval_blank --name e2e_blank
$A paired runs/stage_3/eval runs/ft_imagenet/eval runs/ft_imagenet_fp/eval --name vs_stage3
$A trips runs/stage_3/eval_repro runs/ft_imagenet/eval runs/ft_imagenet_fp/eval
$A gradcam runs/stage_3/best.pt runs/ft_imagenet/best.pt runs/ft_imagenet_fp/best.pt
.venv/bin/python -I scripts/error_geography.py runs/stage_3 runs/ft_imagenet runs/ft_imagenet_fp --out results/final/error_geography.json
.venv/bin/python -I scripts/final_figures.py
.venv/bin/python -I scripts/final_extra.py
.venv/bin/python -I scripts/knn_probe.py runs/ft_imagenet runs/ft_imagenet_fp --out results/final/knn_probe_e2e.json
.venv/bin/python -I scripts/contributor_probe.py runs/ft_imagenet runs/ft_imagenet_fp
cp results/final/figures/gradcam_panel.* report/final/figures/
