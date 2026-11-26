# Instructions for Claude on the Mac

You are picking up a finished project on the Apple M5 laptop, where the end-to-end runs were trained.
The report (`report/final/main.tex`, 19 pages; `report/final_bam/main.tex`, 12 pages) is complete. What
remains is a small amount of extra training that makes its central claim sturdier. Read this whole file
first, then ask the user before starting any run longer than about 30 minutes.

## Why more training

The headline result is that fingerprint removal shrinks the photographer gap from +5.2 pp to +3.1 pp
(paired change -2.1 pp [-3.3, -1.1]) at a cost of 3.6 pp of test accuracy. It rests on **one pair of
runs with one seed** (`runs/ft_imagenet`, `runs/ft_imagenet_fp`), and those runs were disturbed: planned
for 8 epochs, they swapped, and were resumed with a cosine schedule shortened to 6 epochs, so the
learning rate dropped in one step from about 0.0062 to 0.0034 at epoch 4.
See "Training configuration" and "Limitations" in the report.

## Tasks, in priority order

### 1. Clean repeat of the standard / fingerprint-removal pair, new seeds (about 1 to 2 h in total)

- Train `ft_imagenet` and `ft_imagenet_fp` again with `--seed 1` (and `--seed 2` if time allows),
  **one run at a time** (running both at once made the machine swap).
- Use exactly the end-to-end configuration of the report's Table "Training configuration", column (5):
  all layers trainable, SGD Nesterov 0.9, batch 256, lr **0.01** per 256 for all layers (head multiplier
  1), 1 epoch warm-up, cosine, weight decay 1e-4, dropout 0.2, label smoothing 0.1, random resized crop
  0.6 to 1, no flips, bf16. Do **not** use the script default lr (0.02, head x10): it diverges.
  Use a single uninterrupted schedule, 6 epochs, so the LR step is gone and the epoch count matches.
- Fingerprint arm adds `--fingerprint` (bottom 20% blanked, JPEG q30 to 95 with p 0.8, white balance,
  brightness, contrast, saturation jitter, no hue jitter; no `--static-mask`).
- Output to `runs/ft_imagenet_s1`, `runs/ft_imagenet_fp_s1`, and so on. Never overwrite the original
  `runs/ft_imagenet*` folders.
- Evaluate each with `scripts/evaluate.py --ckpt runs/<run>/best.pt --features`.
  Select `best.pt` by val balanced accuracy, as before.
- Compare with `scripts/compare_runs.py` (paired bootstrap): baseline `runs/ft_imagenet_s1/eval`
  against `runs/ft_imagenet_fp_s1/eval`, per seed. Compare to `results/final/paired_fp_vs_standard.json`
  for the output format.

### 2. Wider linear-probe refit (7 min, CPU is fine)

`scripts/linear_probe.py` chose its largest grid value (L2 = 0.01), so the optimum is at the edge.
Widen the grid upward (for example 0.01 to 1), refit, re-evaluate, and update the (2) rows in the
report's tables. Run only if it takes about 10 minutes.

### 3. Optional: the from-scratch pair (4 to 5 h per run, ask first)

`--init scratch` with and without `--fingerprint`, 60 epochs, lr 0.2 per 256, weight decay 5e-4,
3 warm-up epochs (README, section 3). This is the one planned arm of the 2x2 design that was never run.
Only do this if the user says they have the time.

## Things to check before you start

- `scripts/train.py` in this repo selects `cuda` or `cpu` only (see its `device` line). The M5 runs
  must have used a locally patched copy that selects `mps`, and that patch may not be committed. Look
  at `git status` and `git stash list`. If the patch is missing, add the smallest change that picks
  `mps`, keep bf16 autocast, and tell the user what you changed. Do not change the training logic.
- `data/c32/` is not in git except where noted. `scripts/restore_manifest.py` documents how the
  manifest was rebuilt on this machine: the evaluation images are identical to the progress report, and
  the training split is a statistically equivalent re-draw. Check that `data/c32/manifest.csv`,
  `images.bin` and `classes.json` exist before training. Use the same manifest for all new runs.
- Confirm that the original checkpoint still reproduces the committed predictions: run
  `scripts/evaluate.py` on `runs/stage_3/best.pt` and compare with `runs/stage_3/eval/predictions.csv`.
  The report says all 38,118 predictions matched exactly on this machine.
- Keep the laptop usable: one run at a time, no heavy parallel jobs, and plug it in.

## Updating the report afterwards

Only after the new numbers exist:

1. In `report/final/main.tex`, update the abstract, Table "Model comparison", "Removing the camera
   fingerprint" (section "During training") and "Limitations". Report the new effect per seed and
   across seeds. Keep the original pair, labelled as the first run, and say the schedule differed.
   If the effect does not replicate, say so plainly. Do not choose the favourable seed.
2. Mirror the changes in `report/final_bam/main.tex`.
3. Recompile with `tectonic main.tex` in each folder and commit the PDFs.
4. Save new JSON results under `results/final/` and update `scripts/final_extra.py` if it needs the
   new run names.
5. Add a short "End-to-end runs and fingerprint ablation" section to `README.md` if it is not there yet.
   The README currently presents stage 3 as final.

## Rules

- Do not push or open a pull request without asking the user.
- Do not delete or overwrite any existing run folder or committed result.
- The OSV-5M data are CC-BY-SA-4.0. Do not commit images, `images.bin` or checkpoints larger than
  about 50 MB.
- Report outcomes plainly, including runs that fail or results that contradict the report.
