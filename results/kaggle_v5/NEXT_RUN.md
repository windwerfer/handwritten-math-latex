# Proposed next Kaggle run (after quota reset)

Weekly GPU quota reset at **Saturday 28 Nov 2026, 01:00 Europe/Berlin** (30 h fresh).

**v6 (launching): continue v5 from v5-best** for more epochs.
The v5 mixed run stopped at 4 of 10 planned epochs and was still improving
(valid subset exact 68.5% → 72.3% → 75.0% → 76.0%; full valid greedy 73.98%).

## Config

- start: models/v5best (uploaded as `v5best__*` on `windwerfer/handwriter-texo-checkpoints`)
- `--color-strokes --color-mix 0.70 --human-image-prob 0.40 --eval-plain-images`
- 1:1 human/synthetic, batch 32, beams 1 4, `--final-full-valid`
- **LR: fresh cosine, peak 1.5e-5, warmup 300 steps.** v5 used peak 3e-5 over a
  budget-fitted 4-epoch cosine, so its last logged LR was ~4e-12. A new
  `train.py` process does not reload the optimiser; v6 therefore starts a new
  schedule at half of v5's peak (standard continue-FT, avoids an LR shock)
  and lets the cosine cover this 5-epoch / time-capped horizon. See
  `kaggle/v6_kernel/README.md`.
- planned **5 epochs** (v5 was 95.4 min/epoch, so ~8.0 h training). Time cap
  `min(session leftover, 8.30 h)` with a 0.35 h reserve for colour + grey
  InkML + deepcopy evals (greedy and beam 4) plus full valid greedy. Skip
  ONNX in the kernel. Smoke gate first.
- Kernel: `kaggle/v6_kernel` (`windwerfer/handwriter-texo-v6-continue-v5`)
- keep the new checkpoint only if official-test colour, grey InkML, and deepcopy
  beam-4 scores are all at least as good as v5-best (colour 63.91% / 6.95% / 5.16%,
  grey 63.80% / 6.92% / 5.18%, photos 63.72% / 6.96% / 5.21%)

After that (or instead, if the continue-run plateaus): roadmap step 5
(distillation from TexTeller 3) and step 6 (hard-case oversampling).
