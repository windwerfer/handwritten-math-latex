# TODO: improving the model

Goal: beat ink2latex by at least 10% relative on the full MathWriting test set
(target: >= 63.3% exact, TER <= 5.56%). Size budget: 35-40 MB, up to ~50 MB if clearly better.
Baseline v2 (beam 4): 57.7% exact, CER 8.73%, TER 6.66%.
We work down the list and keep a step only if it improves the full valid/test score.
Gains are rough estimates in exact-match points, not measurements.

| # | Step | Est. gain | Cost | Status |
|---|------|-----------|------|--------|
| 1 | v3: MathWriting synthetic data + stroke augmentation | +2 to +5 | ~9.5 GPU-h | running |
| 2 | Decoding: length-normalised beam, brace/grammar-constrained decoding, checkpoint averaging (EMA/SWA) | +0.5 to +2 | CPU only | todo |
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
- Licence: TexTeller is Apache-2.0, so distilling from it is clean; NC data (MathWriting) stays the limiting licence.
