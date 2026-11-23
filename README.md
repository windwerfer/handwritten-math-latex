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
