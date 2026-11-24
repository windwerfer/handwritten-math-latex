# TODO: improving the model

Goal: beat ink2latex by at least 10% relative on the full MathWriting test set
(target: >= 63.3% exact, TER <= 5.56%). Size budget: 35-40 MB, up to ~50 MB if clearly better.
Baseline v2 (beam 4, HF `length_penalty=1.0`): 57.7% exact, CER 8.73%, TER 6.66%. After step 2, recommended decode is beam 4 with `length_penalty=0.0`: 57.8% exact, CER 8.60%, TER 6.58% (see results/decode_v2/SUMMARY.md).
**Current best: v3** (beam 4, `length_penalty=0.0`): 62.55% exact, CER 7.39%, TER 5.49% (results/decode_v3/SUMMARY.md). TER target met; exact match still 0.75 points short of 63.3%.
We work down the list and keep a step only if it improves the full valid/test score.
Gains are rough estimates in exact-match points, not measurements.

| # | Step | Est. gain | Cost | Status |
|---|------|-----------|------|--------|
| 1 | v3: MathWriting synthetic data + stroke augmentation | measured **+4.7 exact** (beam 4 lp0: 57.8% → 62.55%; CER 8.60% → 7.39%, TER 6.58% → 5.49%) | 8.3 GPU-h (1× T4, 6 epochs, best epoch 4) | **done** |
| 2 | Decoding: length-normalised beam, brace/grammar-constrained decoding, checkpoint averaging (EMA/SWA) | v2: +0.13 exact (lp0 57.81% vs lp1 57.68%); v3: tie (lp0 62.55% / CER 7.39% / TER 5.49% vs lp1 62.56% / 7.43% / 5.56%); averaging v3 best+last: no gain (valid 71.6% vs 71.7%). Keep `length_penalty=0.0`; brace no-op on v2 | CPU only | **done** (v2 + v3) |
| 3 | Stroke info in the image: encode stroke order/direction/time as colour channels (cheap "strokes" without a new encoder) | +1 to +4 | 1 run | todo |
| 4 | Distillation from TexTeller 3 (Apache-2.0 weights) as teacher: pseudo-labels + soft targets on synthetic/unlabelled ink | +2 to +6 | teacher inference + 1-2 runs | todo |
| 5 | Hard cases: oversample case/look-alike errors (w/omega, v/nu, d/partial, . vs \cdot), rare tokens, long formulas | +1 to +2 | 1 run | todo |
| 6 | Bigger model within ~50 MB fp16 (~25M params: wider decoder / higher input resolution) | +1 to +3 | 1-2 runs | todo |
| 7 | More synthetic data: arXiv/Wikipedia formulas composed from MathWriting symbol strokes | +1 to +3 | data work + 1 run | todo |
| 8 | Real stroke encoder fused with the image encoder (shared decoder; images-only still works) | +2 to +5 | several runs | todo |
| 9 | Final: quantisation check (fp16 / fp16+int8), ONNX parity, release v-next weights on Hugging Face | keeps size | CPU | todo |

Notes
- Gains overlap; they do not simply add up.
- Kaggle: ~30 GPU-h/week (resets Saturday), max 2 sessions; one run per idea, smoke gate first.
- Official test set stays fixed; compare on full test with hw/normalize.py.
- Step 2 details: results/decode_v2/ and results/decode_v3/. Averaging only two adjacent epochs did not help; try EMA or more snapshots in a later run.
- Licence: TexTeller is Apache-2.0, so distilling from it is clean; NC data (MathWriting) stays the limiting licence.
