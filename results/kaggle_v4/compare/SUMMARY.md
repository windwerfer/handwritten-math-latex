# v4 fairness and robustness (official test, 7,644, beam 4, length_penalty=0.0)

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

v4 colour vs v3 grey InkML (fairest stroke-to-stroke comparison): **+1.03 exact points** (4,891 vs 4,812), CER 7.25% → 6.98%, TER 5.44% → 5.15%. The ink2latex 10% relative target (≥63.3% exact, TER ≤ 5.56%) is met on colour InkML. v4 stopped after 3 of 10 planned epochs and was still improving (valid subset EM 0.6775 → 0.7305 → 0.751; best epoch 2). Full valid greedy 73.13% (n=15,674).

v4 on our own greyscale InkML renders stays strong (only −0.47 exact vs colour). The failure mode is **official pre-rendered / photo-like images**, not greyscale as such: predictions often insert extra dots and stray letters (texture read as stroke-time/direction). A mixed follow-up must therefore include deepcopy raster images (`--human-image-prob > 0`), not only grey InkML (`--color-mix < 1`).

v5 plan (from v4-best, one T4, skip ONNX in kernel): `--color-strokes --color-mix 0.70 --human-image-prob 0.40 --eval-plain-images`. Of a 1:1 human/synthetic epoch that is about 56% colour ink, 24% grey ink, 20% official raster images. Weekly GPU remaining at planning time: 8.08 h (refresh Sat 2026-11-28 01:00 Europe/Berlin); leave ~30 min margin.

Predictions: `ink_grey/`, `deepcopy_grey/`, and `../run/predictions_finetuned_beam{1,4}.csv`.

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
