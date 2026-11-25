# v5 kernel launch

Pushed: 2026-11-25 08:07 Europe/Berlin
Slug: windwerfer/handwriter-texo-v5-mixed-colour-grey
Version: 1
Status at push: RUNNING (not queued)
URL: https://www.kaggle.com/code/windwerfer/handwriter-texo-v5-mixed-colour-grey

Config:
- start: models/v4best (from dataset windwerfer/handwriter-texo-checkpoints)
- --color-strokes --color-mix 0.70 --human-image-prob 0.40 --eval-plain-images
- 1:1 human/synthetic, human_image_prob applies to human samples
- epoch mix: ~56% colour ink, ~24% grey ink, ~20% official raster images
- epochs: 10, stopped by time cap; warmup 200; batch 32; lr 3e-5; beams 1 4; final full valid
- ONNX export skipped in kernel
- WEEKLY_LEFT_H=8.08, MARGIN_H=0.50, eval_reserve_h=0.35
- training budget after prep/smoke: weekly leftover minus margin minus eval reserve
- expected training ~6.9 h (~3–4 epochs at v4's 113.7 min/epoch)
- expected total GPU session ~7.58 h

ETA: 2026-11-25 15:41 Europe/Berlin (about 15:42)

Repo: https://github.com/windwerfer/handwritten-math-latex
Commits: 581033b (results/docs/kernel), bd8ad4e (budget reserve fix, HEAD)
