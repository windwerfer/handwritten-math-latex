# Fairness and robustness (official test, 7,644, beam 4, length_penalty=0.0)

Includes v5 mixed colour/grey/photo results.

All rows use the same official MathWriting test labels and `hw/normalize.py`.
Scored on the box CPU on 25 Nov 2026 (Europe/Berlin), except v4 colour InkML
which is reused from the Kaggle v4 kernel CSVs (rescored locally; counts match).

| model | input | CER | TER | exact match | exact / 7644 | notes |
|---|---|---|---|---|---|---|
| v3 best | deepcopy greyscale (published) | 7.39% | 5.49% | 62.55% | 4,781 | results/decode_v3/; photos/scans domain |
| v3 best | greyscale InkML | 7.25% | 5.44% | 62.95% | 4,812 | 995 s; fair stroke-render protocol |
| v4 best | colour InkML (R=time, G=dir, B=ink) | 6.98% | 5.15% | **63.98%** | 4,891 | native v4 domain; also greedy 63.29% (4,838), CER 7.26%, TER 5.41% |
| v4 best | greyscale InkML (replicated to 3ch) | 7.19% | 5.32% | 63.51% | 4,855 | 896 s; still +0.56 vs v3 on the same renders |
| v4 best | deepcopy greyscale photos | 46.44% | 43.94% | 25.30% | 1,934 | 1081 s; domain collapse — do not ship v4 for photos |
| v5 best (mixed, 4 epochs, still improving) | colour InkML | 6.95% | 5.16% | 63.91% | 4,885 | −0.07 exact vs v4 colour (tie); greedy 63.58% (4,860), CER 7.16%, TER 5.34% |
| v5 best | greyscale InkML | **6.92%** | 5.18% | 63.80% | 4,877 | **best grey InkML**; +0.29 vs v4, +0.85 vs v3; greedy 63.47% (4,852) |
| v5 best | deepcopy greyscale photos | 6.96% | 5.21% | 63.72% | 4,871 | **recovers photos** and beats v3 by +1.17 exact; greedy 63.20% (4,831) |

v4 colour vs v3 grey InkML (fairest stroke-to-stroke comparison): **+1.03 exact points** (4,891 vs 4,812), CER 7.25% → 6.98%, TER 5.44% → 5.15%. The ink2latex 10% relative target (≥63.3% exact, TER ≤ 5.56%) is met on colour InkML. v4 stopped after 3 of 10 planned epochs and was still improving (valid subset EM 0.6775 → 0.7305 → 0.751; best epoch 2). Full valid greedy 73.13% (n=15,674).

v4 on our own greyscale InkML renders stays strong (only −0.47 exact vs colour). The failure mode is **official pre-rendered / photo-like images**, not greyscale as such: predictions often insert extra dots and stray letters (texture read as stroke-time/direction). A mixed follow-up must therefore include deepcopy raster images (`--human-image-prob > 0`), not only grey InkML (`--color-mix < 1`).

v5 (from v4-best, one T4, skip ONNX in kernel) used `--color-strokes --color-mix 0.70 --human-image-prob 0.40 --eval-plain-images`. Of a 1:1 human/synthetic epoch that is about 56% colour ink, 24% grey ink, 20% official raster images. It trained 4 of 10 planned epochs (5.69 h of training, 7.01 h cap, ~95.4 min/epoch) and was still improving (valid subset exact 68.5% → 72.3% → 75.0% → 76.0%; best epoch 3). Full valid greedy 73.98% (n=15,674).

v5 is a near-tie with v4 on colour strokes (−6 exact, CER slightly better) and is the best model on greyscale InkML and on deepcopy photos. All three v5 beam-4 protocols meet the ink2latex target (≥63.3% exact, TER ≤ 5.56%). **Shipped default is v5**: one model for strokes and for photos/scans. Do not ship v4 for photos. v4 remains a colour-stroke specialist (4,891 vs 4,885 exact) if a strokes-only build is ever needed.

Predictions: v4 in this directory (`ink_grey/`, `deepcopy_grey/`, `../run/`); v5 in `results/kaggle_v5/run/predictions_{finetuned,grey_ink,deepcopy}_beam{1,4}.csv`. Details also in `results/kaggle_v5/compare/SUMMARY.md`.

## ONNX export (v4 best, colour InkML)

Exported locally with `scripts/export_onnx.py` (fp16, and fp16-enc + int8-dec are the ship options).
Parity: **200 random official-test colour InkML renders**, seed 0, greedy, box CPU.
Metrics in `results/kaggle_v4/onnx_parity.json`. Graphs are gitignored (`*.onnx`).

| variant | encoder + decoder size | CER | exact match | identical to PyTorch | s/img |
|---|---|---|---|---|---|
| PyTorch fp32 | – | 8.72% | 63.5% | – | 0.066 |
| ONNX fp32 | 57.3 + 22.7 = 80.0 MB | 8.72% | 63.5% | 100% | 0.044 |
| ONNX fp16 | 28.7 + 11.4 = 40.2 MB | 8.72% | 63.5% | 100% | 0.047 |
| ONNX int8 (dynamic, both) | 14.6 + 5.9 = 20.5 MB | 8.78% | 64.5% | 90% | 0.244 |
| encoder fp32 + decoder int8 | 63.2 MB | 8.72% | 63.5% | 97% | 0.035 |
| **encoder fp16 + decoder int8** | **34.6 MB** | 8.21% | 63.5% | 97.5% | 0.034 |

fp16 is the safe default (~40 MB, identical to PyTorch on this sample). The fp16 encoder + int8 decoder is the smaller option (~35 MB, 97.5% identical). Full int8 is smaller but slower and less faithful on CPU. This CER is the 200-sample figure, not the full-test 7.26% greedy.
