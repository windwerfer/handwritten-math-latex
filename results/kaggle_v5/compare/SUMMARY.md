# v5 mixed colour/grey/photo (official test, 7,644, beam 4, length_penalty=0.0)

Kernel `windwerfer/handwriter-texo-v5-mixed-colour-grey` version 1, COMPLETE
(started 08:07, finished ~14:56 Europe/Berlin on 25 Nov 2026). Continues from
v4-best with `--color-strokes --color-mix 0.70 --human-image-prob 0.40
--eval-plain-images`. 1:1 human/synthetic; about 56% colour ink, 24% grey ink,
20% official raster images per epoch. 4 of 10 planned epochs, 5.69 h training
(7.01 h cap, 95.4 min/epoch, 14,366 steps/epoch). Best valid subset is epoch 3
(still improving): exact 68.5% → 72.3% → 75.0% → 76.0% (n=2,000, greedy).
Full valid greedy 73.98% (11,595 / 15,674).

`results/kaggle_v5/run/metrics.json` was verified: every official-test CSV
rescores to the same CER / TER / exact match (see `metrics_verified.json`).

| model | input | CER | TER | exact match | exact / 7644 | notes |
|---|---|---|---|---|---|---|
| v3 best | deepcopy greyscale | 7.39% | 5.49% | 62.55% | 4,781 | previous photo best |
| v3 best | greyscale InkML | 7.25% | 5.44% | 62.95% | 4,812 | |
| v4 best | colour InkML | 6.98% | 5.15% | 63.98% | 4,891 | colour-stroke specialist; greedy 63.29% (4,838) |
| v4 best | greyscale InkML | 7.19% | 5.32% | 63.51% | 4,855 | |
| v4 best | deepcopy greyscale photos | 46.44% | 43.94% | 25.30% | 1,934 | collapsed; do not ship for photos |
| **v5 best** | colour InkML | 6.95% | 5.16% | 63.91% | 4,885 | −0.07 exact vs v4 (tie); greedy 63.58% (4,860), CER 7.16%, TER 5.34% |
| **v5 best** | greyscale InkML | **6.92%** | 5.18% | 63.80% | 4,877 | best grey InkML; +0.29 vs v4, +0.85 vs v3; greedy 63.47% (4,852), CER 7.12%, TER 5.34% |
| **v5 best** | deepcopy greyscale photos | 6.96% | 5.21% | 63.72% | 4,871 | recovers photos and beats v3 by +1.17; greedy 63.20% (4,831), CER 7.22%, TER 5.42% |

**Shipped default is v5**: one model for stroke input and for photos/scans. All
three v5 beam-4 protocols meet the ink2latex 10% relative target (≥63.3% exact,
TER ≤ 5.56%). Colour-only v4 is statistically a tie (6 more exact formulas)
but unusable on photos.

Next Kaggle run after the weekly GPU reset (Saturday 28 Nov 2026, 01:00
Europe/Berlin): continue from v5-best for more epochs (valid was still rising
at epoch 3). After that, roadmap steps are distillation from TexTeller 3 and
hard-case oversampling. Do not start a run until the quota resets.

## ONNX export (v5 best)

Exported locally with `scripts/export_onnx.py`. Parity: **200 random official-test
samples**, seed 0, greedy, box CPU, on colour InkML and on deepcopy photos.
Metrics in `results/kaggle_v5/onnx_parity.json`. Graphs are gitignored (`*.onnx`).

Colour InkML:

| variant | encoder + decoder size | CER | exact match | identical to PyTorch | s/img |
|---|---|---|---|---|---|
| PyTorch fp32 | – | 9.00% | 62.0% | – | 0.123 |
| ONNX fp32 | 57.3 + 22.7 = 80.0 MB | 9.00% | 62.0% | 100% | 0.077 |
| ONNX fp16 | 28.7 + 11.4 = 40.2 MB | 9.00% | 62.0% | 100% | 0.077 |
| ONNX int8 (dynamic, both) | 14.6 + 5.9 = 20.5 MB | 8.78% | 63.0% | 88.0% | 0.354 |
| encoder fp32 + decoder int8 | 63.2 MB | 8.46% | 62.5% | 95.5% | 0.057 |
| **encoder fp16 + decoder int8** | **34.6 MB** | 8.15% | 63.0% | 95.5% | 0.055 |

Deepcopy photos:

| variant | encoder + decoder size | CER | exact match | identical to PyTorch | s/img |
|---|---|---|---|---|---|
| PyTorch fp32 | – | 7.14% | 60.0% | – | 0.104 |
| ONNX fp32 | 57.3 + 22.7 = 80.0 MB | 7.14% | 60.0% | 100% | 0.073 |
| ONNX fp16 | 28.7 + 11.4 = 40.2 MB | 7.14% | 60.0% | 100% | 0.076 |
| ONNX int8 (dynamic, both) | 14.6 + 5.9 = 20.5 MB | 7.12% | 59.0% | 88.5% | 0.375 |
| encoder fp32 + decoder int8 | 63.2 MB | 6.97% | 59.5% | 97.5% | 0.060 |
| **encoder fp16 + decoder int8** | **34.6 MB** | 7.01% | 59.5% | 97.0% | 0.058 |

fp16 is the safe default (~40 MB, identical to PyTorch on both domains). The
fp16 encoder + int8 decoder is the smaller option (~35 MB; 95.5% identical on
colour, 97.0% on photos). Full int8 is smaller but slower and less faithful.
**Shipped default stays v5** — no parity issue.

## Next Kaggle run (after Saturday 28 Nov 2026, 01:00 Europe/Berlin quota reset)

Do not start a run until the weekly GPU quota resets. Proposed plan:

1. **Continue v5 from v5-best** for more epochs (first use of the new quota).
   Valid subset exact was still rising at epoch 3 (68.5% → 72.3% → 75.0% → 76.0%).
   Keep `--color-strokes --color-mix 0.70 --human-image-prob 0.40 --eval-plain-images`,
   batch 32, lr 3e-5, beams 1 and 4, skip ONNX in the kernel. Size the cap to
   remaining weekly hours minus a ~30 min margin (and reserve time for colour +
   grey + deepcopy evals). Expect ~4–5 more epochs in a ~8–10 h session at
   ~95 min/epoch. Keep the run only if full-test colour, grey InkML, and deepcopy
   all stay at least as good as v5-best (especially colour ≥ 63.3% exact and
   photos not regressing toward v3).
2. After that (or instead, if the continue-run plateaus): **distillation from
   TexTeller 3** (roadmap step 5) and **hard-case oversampling** (step 6).
