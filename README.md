# handwriter — handwritten math → LaTeX (xnotes), first test run

Goal: verify a data + fine-tune + eval pipeline by fine-tuning an existing small model,
not training from scratch. Training in Python/PyTorch; Rust/ONNX inference comes later.

## Choices
- **Model: Texo** (`alephpi/FormulaNet` on HF, code github.com/alephpi/Texo), 20M params:
  HGNetv2 CNN encoder + 2-layer MBart decoder, 384x384 input, 687-token KaTeX vocab.
  Weights + code are **AGPL-3.0** (matters if xnotes ships closed-source).
  Not trained on MathWriting (UniMER-1M; its handwritten part is CROHME/HME100K), so the
  zero-shot vs fine-tuned comparison on MathWriting is meaningful. ONNX export already
  exists in the HF repo (`onnx/encoder_model.onnx`, `onnx/decoder_model_merged.onnx`).
- Rejected for now: `tjoab/latex_finetuned` (TrOCR-base, ~334M params, already trained on
  MathWriting → a fine-tune would only re-validate the pipeline; too large for Android and
  for this box's ~3 GB free RAM). Good as a reference baseline later.
- **Data: Google MathWriting-2024, human inks**, official train/valid/test splits, via the
  pre-rendered HF mirror `deepcopy/MathWriting-human` (parquet, images + normalized LaTeX).
  License **CC BY-NC-SA 4.0** (non-commercial!).

## Layout
- `hw/latex_tok.py` — MathWriting LaTeX ↔ Texo token mapping (e.g. `\infty`↔`\infin`,
  `\rightarrow`↔`\rarr`, `matrix`↔`array`), eval canonicalisation.
- `hw/data.py` — Texo preprocessing (crop margin, fit 384², pad, grayscale-normalise), dataset, collate.
- `hw/model.py` — load Texo via transformers `VisionEncoderDecoderModel`; CE loss computed manually.
- `hw/metrics.py` — CER (whitespace-insensitive chars), token error rate, exact match.
- `hw/texo_vendor/` — HGNetv2 encoder vendored from Texo (AGPL-3.0).
- `scripts/prep_data.py` — subset builder. `scripts/train.py` — zero-shot eval, fine-tune, eval, sample PNGs.
- `kaggle/dataset`, `kaggle/kernel` — private Kaggle dataset + GPU kernel definitions.

## Steps
```bash
uv venv -p 3.11 .venv && . .venv/bin/activate
uv pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers==4.40.0 albumentations opencv-python-headless pillow rapidfuzz pyarrow pandas matplotlib
# data: data/raw/{train-00000-of-00003,val-*,test-*}.parquet from HF deepcopy/MathWriting-human
# model: models/texo/ = config.json, generation_config.json, model.safetensors, tokenizer*.json from HF alephpi/FormulaNet
python scripts/prep_data.py --train 5000 --valid 500 --test 500
# CPU smoke test
python scripts/train.py --out runs/smoke --train-limit 48 --valid-limit 8 --test-limit 20 \
  --epochs 1 --max-steps 12 --batch-size 4 --eval-batch-size 4 --warmup 2 --lr 2e-5 --workers 0 --no-save
# Kaggle (GPU T4)
cd kaggle/dataset && kaggle datasets version -p . -m "update"   # or `create` the first time
cd ../kernel && kaggle kernels push -p .
kaggle kernels status windwerfer/handwriter-texo-mathwriting-finetune
kaggle kernels output windwerfer/handwriter-texo-mathwriting-finetune -p kaggle/output/v1
```

## Notes / gotchas
- Texo labels are whitespace-separated tokens; MathWriting labels are compact → `to_texo()`.
- Texo's own Lightning code shifts labels left and keeps `<s>` in decoder inputs; we mirror that
  and compute the loss ourselves (transformers' VisionEncoderDecoder label shifting differs by version).
- BatchNorm running stats are kept frozen during fine-tuning (small batches wrecked them in the
  smoke test: CER 0.26 → 2.18 after 12 steps with BN in train mode).

## Results — Kaggle v1 (2026-11-23, Tesla T4, 5 epochs, 5000 train imgs, bs 32, lr 5e-5, ~5 min train)
Official MathWriting **test, all 7644 human inks**, greedy decoding:

| | CER | token error rate | exact match |
|---|---|---|---|
| Texo zero-shot | 39.1 % | 29.9 % | 10.0 % |
| Texo fine-tuned (5k train) | **17.8 %** | 13.3 % | **32.5 %** |

Valid (500) CER per epoch: 25.7 → 19.7 → 17.5 → 16.6 → 16.0 %. Final train loss 0.054 → overfitting
on 5k images; more data is the obvious next lever. Zero-shot is partly penalised by label
conventions (e.g. Texo writes `\lim`, `\sin`; MathWriting labels are `lim`, `sin`).
For reference, tjoab/latex_finetuned (TrOCR, 334M, trained on all of MathWriting) reports 14.9 % CER.
Outputs: `results/kaggle_v1/` (metrics.json, predictions_*.csv, samples/*.png, kernel.log);
fine-tuned weights: `kaggle/output/v1/run/model/` (not in git).
Note: pinning transformers==4.40.0 failed on Kaggle (tokenizers wheel build); the run used Kaggle's
preinstalled transformers, which worked since we compute the loss ourselves.


## v2 — full MathWriting human train, canonical labels (2026-11-23/24)

**Label normalization** (`hw/normalize.py`, tests in `tests/test_normalize.py`, 61 passing): one canonical
style for training targets *and* scoring — function names as commands (`lim`→`\lim`, `sin`→`\sin`,
`inf`→`\inf`, …), standard symbol names (`\infin`→`\infty`, `\rarr`/`\to`→`\rightarrow`, `\ge`→`\geq`,
`\le`→`\leq`, `\ne`→`\neq`, …), `pmatrix`/`array{cc}`→`(\begin{matrix}…\end{matrix})`, braced sub/superscripts,
primes as `^{\prime}`, spacing/sizing commands dropped, `\mathrm{}`/`\operatorname{}` unwrapped, no whitespace.
Changes ~7 % of MathWriting labels (6.6–10.6 % per split; 13–17 % counting whitespace-only changes).
Texo-vocab gaps are bridged bijectively in `hw/latex_tok.py` (`\infty`↔`\infin`, `\rightarrow`↔`\rarr`,
`matrix`↔`array`, `\tanh`/`\inf` → `\operatorname{…}`): **0 OOV tokens** and 0 round-trip mismatches over
train shard 0 + valid + test.

**Training (Kaggle, 2×T4, DataParallel, bs 64, lr 5e-5 cosine, BN stats frozen):** all 229,864 human train
inks. Throughput only ~36 img/s (CPU-bound, 4 vCPUs) → 114 min/epoch, budget cut plan to 5 epochs. The run was
**OOM-killed (host RAM) at step 14,250 (epoch 3, ~7 h)**; the best checkpoint (epoch 2) was saved.
Valid CER (2,000 random official-valid samples): ep0 8.36 % → ep1 13.74 % (spike) → ep2 **6.51 %** (EM 67.9 %).
The ep1 spike came with *higher* exact match (61.9 % vs 59.1 %) and TER > CER, i.e. a few long degenerate
outputs rather than general degradation (per-epoch valid predictions are now saved to confirm next time).

**Test (official MathWriting test, all 7,644 human inks, canonical scoring; Kaggle eval kernel
`windwerfer/handwriter-texo-v2-eval`, `results/kaggle_v2_eval/`):**

| model | decoding | CER | TER | exact match |
|---|---|---|---|---|
| Texo zero-shot | greedy | 38.4 % | 28.9 % | 10.9 % |
| v1 (5k subset, 5 ep) | greedy | 17.8 % | 13.5 % | 32.5 % |
| v2 best (full human train, ep 2) | greedy | 9.03 % | 6.91 % | 56.9 % |
| v2 best | beam 4 | **8.73 %** | **6.66 %** | **57.7 %** |

(For reference: tjoab/latex_finetuned, TrOCR 334M trained on MathWriting, reports 14.9 % CER — different
normalization, not strictly comparable.)

Typical v2 errors (token edit ops, greedy): case confusions dominate (`x`↔`X`, `p`↔`P`, `s`↔`S`, `c`↔`C`,
`k`↔`K`, `z`↔`Z`, `v`↔`V`), `.`↔`\cdot`, Greek vs Latin look-alikes (`\omega`→`w`, `\nu`→`v`, `\kappa`→`k`,
`d`→`\partial`, `z`→`2`), and structure errors: brace/subscript insertions/deletions (`{ } _` are the top
inserted/deleted tokens, e.g. `Spec`→`Sp_{ec}`). Many are genuinely ambiguous in handwriting.

**ONNX (exported on the box from v2 best, `scripts/export_onnx.py`, files in `onnx/v2best/`, not in git;
parity on 200 random official-test samples, CPU ORT 1.23, 4 threads):**

| variant | encoder + decoder size | CER | EM | identical to PyTorch | s/img |
|---|---|---|---|---|---|
| PyTorch fp32 | – | 8.74 % | 58.5 % | – | 0.065 |
| ONNX fp32 | 57.3 + 22.7 = 80.0 MB | 8.74 % | 58.5 % | 100 % | 0.045 |
| ONNX fp16 | 28.7 + 11.4 = 40.2 MB | 8.74 % | 58.5 % | 100 % | 0.049 |
| ONNX int8 (dynamic, both) | 14.6 + 5.9 = 20.5 MB | 9.76 % | 55.5 % | 82.5 % | 0.235 |
| enc fp32 + dec int8 | 63.2 MB | 8.80 % | 58.0 % | 96 % | 0.035 |
| **enc fp16 + dec int8** | **34.6 MB** | 8.56 % | 58.5 % | 97 % | 0.035 |

int8 on the CNN encoder (ConvInteger) is both slower and less accurate on CPU; for <30 MB, try static
(calibrated) QDQ quantization of the encoder or fp16 + int8 decoder. Graphs: `encoder.onnx` (pixel_values
f32[B,3,384,384] → f32[B,144,384]), `decoder.onnx` (input_ids i64[B,T], encoder_hidden_states → logits[B,T,687]),
no KV cache (2-layer decoder); greedy loop + `tokenizer.json` vocab → `hw.latex_tok.from_texo` post-processing.

**OOM cause & fix.** The dataset held 230k images as Python `bytes` objects in a pandas object column
(~1.2 GB + object overhead). Forked DataLoader workers (4 persistent train workers + extra eval workers)
touch each object's refcount on access, so copy-on-write pages were gradually duplicated into every worker
until host RAM (31 GB) ran out after ~7 h. Fix (`hw/data.py`): images in one contiguous `uint8` numpy buffer +
offsets, labels / precomputed Texo targets / ids in fixed-width numpy string arrays (no per-sample Python
objects); eval loaders use ≤2 non-persistent workers (`scripts/train.py`). Also recommended next time: single
GPU without DataParallel (v1 single-T4 got 83 img/s vs 36 img/s here) and fewer CPU augmentations.
