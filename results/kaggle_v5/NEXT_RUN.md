# Proposed next Kaggle run (after quota reset)

Weekly GPU quota is exhausted until **Saturday 28 Nov 2026, 01:00 Europe/Berlin**.
Do not start a kernel before then.

**First run after the reset: continue v5 from v5-best** for more epochs.
The v5 mixed run stopped at 4 of 10 planned epochs and was still improving
(valid subset exact 68.5% → 72.3% → 75.0% → 76.0%; full valid greedy 73.98%).

Suggested config (same mix as v5):
- start: models/v5best
- `--color-strokes --color-mix 0.70 --human-image-prob 0.40 --eval-plain-images`
- 1:1 human/synthetic, batch 32, lr 3e-5, warmup 200, beams 1 4, `--final-full-valid`
- epochs 10 with a time cap sized to remaining weekly hours minus ~30 min margin
  and ~0.35 h for colour + grey InkML + deepcopy evals
- skip ONNX export in the kernel
- smoke gate first
- keep the new checkpoint only if official-test colour, grey InkML, and deepcopy
  beam-4 scores are all at least as good as v5-best (colour 63.91% / 6.95% / 5.16%,
  grey 63.80% / 6.92% / 5.18%, photos 63.72% / 6.96% / 5.21%)

At v5's 95.4 min/epoch, an ~8–10 h session is about 4–5 further epochs.

After that (or instead, if this continue-run plateaus): roadmap step 5
(distillation from TexTeller 3) and step 6 (hard-case oversampling).
