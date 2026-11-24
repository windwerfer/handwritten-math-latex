# Competitor comparison: handwriter v2 vs ink2latex r3 vs TeXPen / TexTeller3

Evaluated 24 Nov 2026, Europe/Berlin. All scored with the same `hw.normalize` canonicalisation and the same metrics (exact match, CER, TER). No numbers below are copied from the competitors' READMEs except where explicitly labelled "published".

Primary like-for-like set: a **fixed random 1,000-sample subset** of the official MathWriting test (seed 0, IDs in `subset_1000.json`). ink2latex was also run on the **full 7,644** official test (greedy and beam 5). TexTeller was **not** run on the full 7,644 (about 2.8 h at 1.3 s/image on this CPU).

## Like-for-like table (our scoring)

| model | input | size (ship) | licence | n | decode | exact | CER | TER |
|---|---|---|---|---|---|---|---|---|
| **handwriter v2 best** | image | 40.2 MB fp16 (34.6 MB enc-fp16+dec-int8) | planned MIT code; weights inherit MathWriting NC | 1000 | greedy | 56.3% | 8.44% | 6.47% |
| **handwriter v2 best** | image | same | same | 1000 | beam 4 | 57.2% | 8.21% | 6.32% |
| **handwriter v2 best** | image | same | same | **7644** | greedy | 56.9% | 9.03% | 6.91% |
| **handwriter v2 best** | image | same | same | **7644** | beam 4 | **57.7%** | **8.73%** | **6.66%** |
| **ink2latex r3** | strokes | 18.8 MB `r3-ink.bin`; 35.0 MB ONNX zip; 37.6 MB `.pt` | PolyForm NC code; CC BY-NC-SA 4.0 weights | 1000 | greedy | 56.5% | 7.69% | 5.84% |
| **ink2latex r3** | strokes | same | same | 1000 | beam 5 | 57.2% | 7.35% | 5.75% |
| **ink2latex r3** | strokes | same | same | **7644** | greedy | 58.5% | 8.66% | 6.44% |
| **ink2latex r3** | strokes | same | same | **7644** | beam 5 | **59.2%** | **8.28%** | **6.18%** |
| **TeXPen default / TexTeller3 ONNX int8** | image | 317.6 MB (enc 88.3 + dec 229.3) | Apache-2.0 code and claimed Apache-2.0 weights | 1000 | greedy | **74.5%** | 7.25% | **4.75%** |

v2 full-test numbers are the existing Kaggle eval (`results/kaggle_v2_eval/`), already scored with `hw.normalize`. ONNX fp16 had 100% greedy parity with that PyTorch checkpoint, so the quality comparison uses those predictions.

## What "10% better" would mean

Two readings; both should be measured against the **stronger** competitor on that metric.

**A. 10% relative fewer errors** (the more standard reading):

| baseline | metric | baseline value | target (10% fewer errors) |
|---|---|---|---|
| v2 beam 4, full 7644 | CER | 8.73% | 7.86% |
| v2 beam 4, full 7644 | TER | 6.66% | 5.99% |
| v2 beam 4, full 7644 | exact errors | 42.3% (exact 57.7%) | exact **61.9%** |
| ink2latex r3 beam 5, full 7644 | CER | 8.28% | 7.45% |
| ink2latex r3 beam 5, full 7644 | TER | 6.18% | 5.56% |
| ink2latex r3 beam 5, full 7644 | exact errors | 40.8% (exact 59.2%) | exact **63.3%** |
| TexTeller3 int8 greedy, subset 1000 | CER | 7.25% | 6.53% |
| TexTeller3 int8 greedy, subset 1000 | TER | 4.75% | 4.28% |
| TexTeller3 int8 greedy, subset 1000 | exact errors | 25.5% (exact 74.5%) | exact **77.1%** |

Our re-score of r3 beam 5 on all 7,644 with `hw.normalize` is **59.2% exact / 8.28% CER / 6.18% TER** — essentially identical to their published 59.2% exact / 6.2% TER. Their native token exact on this run was 59.20% and native TER 6.23%.

**B. +10 points exact match:** v2 57.7% → 67.7%; ink2latex ~59% → ~69%; TexTeller 74.5% → 84.5%. This is a much higher bar.

On the 1,000-subset, ink2latex is **tied with v2 on exact** (both 57.2% at beam 4/5) and only modestly better on CER/TER (~10% relative fewer CER errors than v2: 7.35% vs 8.21%). TexTeller is in a different class on exact (+17 points, 40% relative fewer exact-errors) and clearly better on TER (4.75% vs 6.32%), while CER is only slightly better than ink2latex (7.25% vs 7.35%) because **when TexTeller is wrong, it is more wrong** (CER on misses 26.0% vs 15.7% ink2latex / 17.9% v2): leftover typographic junk such as `\hbox`, `\phantom{\rule...}`, `\boxed{\begin{array}`.

**Beating both by 10% (reading A) means roughly 77% exact or 6.5% CER / 4.3% TER, i.e. beating TexTeller, not just ink2latex.** A 20M image model is unlikely to close that exact-match gap unless v3 (or a later stroke model) is a large jump. Distilling TexTeller, or training a stroke encoder, is more plausible than hoping the current Texo fine-tune path gets there on its own.

On the official full test, ink2latex r3 beam 5 is slightly ahead of v2 beam 4 (+1.5 exact points; 5% relative fewer CER errors, 7% fewer TER errors) — the same league, not 10% better either way. TexTeller is the quality leader on exact and TER, but at 318 MB it is far outside the 35–50 MB ship budget.

## Project facts

### Our model (handwriter / Texo fine-tune)
- Image encoder–decoder, ~20M params, official MathWriting images (and, in v3, synthetic + stroke augmentation).
- Ship size 34.6–40.2 MB ONNX, well inside the 35–40–50 MB budget.
- Trained only on MathWriting (CC BY-NC-SA 4.0), so weights cannot be Apache/MIT.

### ink2latex (aburousan/ink2latex)
- Stroke transformer: d=256, 8 heads, 6 encoder + 4 decoder layers, 9,818,920 params, 243 LaTeX tokens, max 1,600 points / 160 tokens.
- Input is ink strokes (photos are traced back into strokes). This is a real advantage for xnotes, which already has strokes.
- Trained from scratch 80k steps on MathWriting train/synthetic/symbols plus the author's own notes and some generated formulas (DATA.md). `valid` chose the checkpoint; `test` was held out.
- **Published** (their tokenisation, beam 5, full MathWriting test): 59.2% exact / 6.2% TER. Our re-score of the same r3 checkpoint, beam 5, all 7,644, with `hw.normalize`: **59.2% exact / 8.28% CER / 6.18% TER** (their native exact 59.20%, native TER 6.23%). Greedy: 58.5% exact / 8.66% CER / 6.44% TER. Their "exact" is token-list equality after their `canonical()` brace normaliser; they also report `loose_exact` (ignores `{`, `}`, `\left`, `\right`). The two normalisers barely differ on this data.
- Licence: PolyForm Noncommercial 1.0.0 code; weights CC BY-NC-SA 4.0. `demo/features.js` and `demo/photo.js` are also MIT, so an app can feed the model under MIT, but the weights stay NC.
- Release assets (v1.0, 7 Nov 2026): `r3.pt` 37.6 MB, `r3-onnx.zip` 35.0 MB, `r3-ink.bin` 18.8 MB (fp16 compact). Same files exist for r3-photo.
- Browser demo: https://aburousan.github.io/ink2latex/

### TeXPen (TeXPen/TeXPen.github.io, 222 stars)
- No README in the repo. It is a TypeScript/React website (Apache-2.0) that runs **Ji-Ha/TexTeller3-ONNX-dynamic**, an ONNX export of **OleehyO/TexTeller** (TexTeller 3.0).
- Architecture: ViT-B/16 encoder (448×448, 1 channel, 12 layers, hidden 768) + TrOCR-style decoder (12 layers, d=1024, 16 heads, vocab 15,000). **298M parameters.**
- Runtime: `@huggingface/transformers` (transformers.js) + `onnxruntime-web`. Default profile is **int8 encoder + int8 decoder, greedy** (`NUM_BEAMS: 1`), ~313 MB download (measured 317.6 MB ONNX). Other profiles: fp32 ~1.25 GB, fp16 enc + fp32 dec, int4 ~224 MB. Still far above a 35–50 MB on-device budget.
- Input: **image only** (canvas is rasterised). TeXPen preprocess: binarise opaque images, trim the ink box, letterbox with the HF `Resize(size=447, max_size=448)` rule, top-left pad on white, then mean/std 0.9545467 / 0.15394445. Our eval matches that. Display wrappers `\[ \]` are stripped before scoring (TexTeller usually emits them; MathWriting labels do not).
- Training data: Tex80M (OleehyO/latex-formulas-80M, Apache-2.0, mostly rendered arXiv formulas) mixed with a handwritten subset. The dataset card warns: *"The handwritten subset of this dataset was collected entirely from existing open source work, which includes all test sets."* We did not independently measure overlap with the MathWriting test, so the 74.5% exact may be inflated by leakage. `handwritten_online` has 8,349 rows, uncomfortably close to the 7,644 MathWriting test rows.
- Published scores are from arXiv:2508.09220, **"Towards Scalable Training for Handwritten Mathematical Expression Recognition"**. That paper is **withdrawn** as of v4 (10 Jan 2026): "temporarily withdraw … to make substantial revisions." The v3 HTML still reports TexTeller at **81.0% ExpRate / 94.3% EPMR / 84.9% EP@0** on MathWriting. ExpRate is command-level exact match; EPMR/EP@0 are their image-based render metrics, not string CER/TER. Our 74.5% exact (int8, greedy, subset, our normalise) is in the same ballpark, not 81%.
- Weights are labelled Apache-2.0. If the handwritten mix includes MathWriting (CC BY-NC-SA) or other NC/test sets, that claim is legally messy.

### `ink2math` name
Not free of collisions, but weaker than `ink2latex`:
- GitHub user/org `ink2math` does not exist.
- Existing repos: `gokuls-bit/Ink2Math` (empty, 1 star), `Ananya0907/Ink2Math` (student CNN–LSTM on CROHME, no MathWriting scores), `AdityaThote27/ink2math-hmer`, `VivekGujar45/INK2MATH_smart_equation_solver`. GitHub repo names are case-insensitive, so `windwerfer/ink2math` would not conflict with those accounts, but searchers will still find the others.
- Hugging Face has no model or dataset named ink2math.
- `handwritten-math-latex` remains the cleaner public name.

## Caveats
- TexTeller was evaluated as the **shipped TeXPen default** (int8, greedy), not fp32 "highest quality". fp32 might be a bit better and is much larger.
- The merged-ONNX KV-cache path diverged after the first token in ONNX Runtime here; we used a correct but slower full-prefix greedy decode (1.30 s/image). Qualitatively checked: exact matches are genuine formula strings, not an unwrap bug.
- Official MathWriting test *images* were already local (`test_full.parquet`). Strokes were extracted from the official `mathwriting-2024.tgz` test split (7,644 InkML files). Weights live under `models/competitors/` (gitignored).
- "Use their project in xnotes" is not a size-neutral choice: TexTeller is ~8× the 40 MB budget; ink2latex fits and is stroke-native, but NC-licensed and only tied with us on exact match.

## Scripts
`scripts/eval_competitors.py` reproduces the comparison (`--score-v2`, `--ink2latex`, `--texteller`, `--subset 1000 --seed 0` or `--full`).
