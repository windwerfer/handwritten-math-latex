# TODO: improving the model

Goal: beat ink2latex by at least 10% relative on the full MathWriting test set
(target: >= 63.3% exact, TER <= 5.56%). Size budget: 35-40 MB, up to ~50 MB if clearly better.
Baseline v2 (beam 4, HF `length_penalty=1.0`): 57.7% exact, CER 8.73%, TER 6.66%. After step 2, recommended decode is beam 4 with `length_penalty=0.0`: 57.8% exact, CER 8.60%, TER 6.58% (see results/decode_v2/SUMMARY.md).
**Current best (strokes):** v4 (colour InkML, beam 4, `length_penalty=0.0`): 63.98% exact (4,891 / 7,644), CER 6.98%, TER 5.15% (results/kaggle_v4/compare/SUMMARY.md). The ink2latex exact and TER targets are met on this protocol. **Current best (photos/scans):** v3 (deepcopy greyscale, same decode): 62.55% exact, CER 7.39%, TER 5.49% (results/decode_v3/SUMMARY.md). Do not use v4 on photos (25.30% exact).
We work down the list and keep a step only if it improves the full valid/test score.
Gains are rough estimates in exact-match points, not measurements.

| # | Step | Est. gain | Cost | Status |
|---|------|-----------|------|--------|
| 1 | v3: MathWriting synthetic data + stroke augmentation | measured **+4.7 exact** (beam 4 lp0: 57.8% → 62.55%; CER 8.60% → 7.39%, TER 6.58% → 5.49%) | 8.3 GPU-h (1× T4, 6 epochs, best epoch 4) | **done** |
| 2 | Decoding: length-normalised beam, brace/grammar-constrained decoding, checkpoint averaging (EMA/SWA) | v2: +0.13 exact (lp0 57.81% vs lp1 57.68%); v3: tie (lp0 62.55% / CER 7.39% / TER 5.49% vs lp1 62.56% / 7.43% / 5.56%); averaging v3 best+last: no gain (valid 71.6% vs 71.7%). Keep `length_penalty=0.0`; brace no-op on v2 | CPU only | **done** (v2 + v3) |
| 3 | Stroke info in the image: encode stroke order/direction/time as colour channels (cheap "strokes" without a new encoder) | measured **+1.03 exact** on colour InkML vs v3 on the same grey InkML (62.95% → 63.98%; CER 7.25% → 6.98%, TER 5.44% → 5.15%). v4 on grey InkML still 63.51%. v4 on deepcopy photos **collapses to 25.30%** (CER 46.44%). Stopped at 3 of 10 epochs and was still improving (valid subset EM 67.8% → 73.1% → 75.1%; full valid greedy 73.13%). | ~7 GPU-h kernel (5.25 h training, 1× T4, from v3best) | **done** (stroke model only; do not use on photos) |
| 4 | Mixed colour/grey training so one model reads images and strokes: continue from v4-best with colour InkML, grey InkML (`--color-mix 0.70`) and official raster images (`--human-image-prob 0.40`); eval colour + grey InkML + deepcopy | recover photo exact toward v3 (~62.5%) without dropping colour-stroke exact below 63.3% | remaining weekly GPU (~7.5 h cap after 30 min margin; skip ONNX in kernel) | **starting** (v5 kernel, from v4best) |
| 5 | Distillation from TexTeller 3 (Apache-2.0 weights) as teacher: pseudo-labels + soft targets on synthetic/unlabelled ink | +2 to +6 | teacher inference + 1-2 runs | todo |
| 6 | Hard cases: oversample case/look-alike errors (w/omega, v/nu, d/partial, . vs \cdot), rare tokens, long formulas | +1 to +2 | 1 run | todo |
| 7 | Bigger model within ~50 MB fp16 (~25M params: wider decoder / higher input resolution) | +1 to +3 | 1-2 runs | todo |
| 8 | More synthetic data: arXiv/Wikipedia formulas composed from MathWriting symbol strokes | +1 to +3 | data work + 1 run | todo |
| 9 | Real stroke encoder fused with the image encoder (shared decoder; images-only still works) | +2 to +5 | several runs | todo |
| 10 | Final: quantisation check (fp16 / fp16+int8), ONNX parity, release v-next weights on Hugging Face | keeps size | CPU | todo (v3 and v4 ONNX already checked; fp16 40.2 MB exact match, fp16+int8 ~35 MB) |

Notes
- Gains overlap; they do not simply add up.
- Kaggle: ~30 GPU-h/week (resets Saturday), max 2 sessions; one run per idea, smoke gate first.
- Official test set stays fixed; compare on full test with hw/normalize.py. Always state the input domain: deepcopy greyscale photos, greyscale InkML, or colour InkML (R=time, G=direction, B=ink). See results/v4_stroke_colour_samples/ and results/kaggle_v4/compare/SUMMARY.md.
- Step 2 details: results/decode_v2/ and results/decode_v3/. Averaging only two adjacent epochs did not help; try EMA or more snapshots in a later run.
- Licence: TexTeller is Apache-2.0, so distilling from it is clean; NC data (MathWriting) stays the limiting licence.
