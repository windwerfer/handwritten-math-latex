# Decode tuning (roadmap step 2)

Tune on a fixed valid subset (1,000 formulas, seed 0), then confirm the winner
once on the official MathWriting test set (all 7,644 human inks) with
`hw.normalize` scoring. Scripts are reusable for later checkpoints (v3, …):

```bash
python scripts/eval_decode.py --model results/kaggle_v2/run/best \
    --valid 'data/raw/val-*.parquet' --test 'data/raw/test-*.parquet' \
    --out results/decode_v2 --valid-limit 1000 --seed 0 \
    --length-penalties 0.0 0.6 0.8 1.0 --brace both
python scripts/average_checkpoints.py --out models/swa --discover runs/v3
```

The published v2 beam-4 baseline already used HuggingFace’s default
`length_penalty=1.0` (`score / length**alpha`). Alpha 0.0 ranks by the raw sum
of log-probabilities (no length normalisation). Grammar-constrained decoding
rejects unmatched `}` / `\end{array}` and forces the matching closer instead of
EOS while the stack is open.

Checkpoint averaging was not possible for v2: the Kaggle run was killed mid
epoch 3, only `best/` (epoch 2) was saved, and
`windwerfer/handwriter-texo-checkpoints` contains texo / v1 / v2best, not
per-epoch snapshots of one run. `scripts/average_checkpoints.py` will average
`best/` + `last/` once a later run writes both.

## Valid subset (1,000, seed 0)

| Setting | Exact | CER | TER | Seconds |
|---|---:|---:|---:|---:|
| `beam4_lp0` | **67.50%** | 6.89% | **5.87%** | 43.0 |
| `beam4_lp0_brace` | 67.50% | 6.89% | 5.87% | 44.2 |
| `beam4_lp0.6` | 67.20% | 6.86% | 5.94% | 51.3 |
| `beam4_lp0.6_brace` | 67.20% | 6.86% | 5.94% | 51.4 |
| `beam4_lp0.8` | 67.20% | **6.82%** | 5.91% | 40.0 |
| `beam4_lp0.8_brace` | 67.20% | 6.82% | 5.91% | 42.7 |
| `beam4_lp1` (HF default / published baseline) | 67.00% | 6.87% | 5.96% | 82.5 |
| `beam4_lp1_brace` | 67.00% | 6.87% | 5.96% | 97.1 |

Winner on valid: `beam4_lp0` (no length penalty, no grammar constraint).

The `_brace` runs are **token-identical** to the unconstrained run at every
alpha (1,000 / 1,000). Beam 4 already produced balanced braces and
`\begin{array}` / `\end{array}` on this subset, so the constraint never
changed a hypothesis. It is cheap insurance (a logits processor) and can stay
available for later checkpoints, but it is not a v2 gain.

Where `lp0` and `lp1` disagree (16 of 1,000), `lp0` is exact on 6 that `lp1`
misses and `lp1` is exact on 1 that `lp0` misses. Typical `lp1` mistakes are
over-structuring, which the length penalty encourages because longer
sequences are rewarded: `\sqrt{Hz}` → `\sqrt{H_{z}}`, `e^{-BR}` → `e^{-B_{R}}`,
`Mg(x)` → `M_{g}(x)`, `Z` → `\mathbb{Z}`, `10\%=0.1` → `10^{\circ}/0=0.1`.
The one `lp1` win is a long matrix where `lp0` dropped the overlines.

## Full test (7,644)

| Setting | Exact | CER | TER | Exact count |
|---|---:|---:|---:|---:|
| `beam4_lp0` (confirmed) | **57.81%** | **8.60%** | **6.58%** | **4419** |
| `beam4_lp1` (published v2 beam 4) | 57.68% | 8.73% | 6.66% | 4409 |

`lp0` is exact on 26 formulas that `lp1` misses and misses 16 that `lp1` gets
right (net +10). CER and TER also improve. This is a small but consistent
gain, in the same direction as the valid subset, and costs nothing at train
time.

The published baseline numbers were **not** re-run; they come from
`results/kaggle_v2_eval/eval/metrics.json` and match a rescore of
`predictions_v2best_beam4.csv`. The `lp0` numbers are a new full pass of
`results/kaggle_v2/run/best` on CPU (encoder states streamed per batch so the
~8.4 GB full-test cache does not need to sit in RAM).

## Recommendation

**Use `length_penalty=0.0` for beam search** (now the default in
`hw.model.predict`, `scripts/eval.py` and `scripts/train.py`). Do not turn
grammar-constrained decoding on by default; it did not change v2 output.
Re-run `scripts/eval_decode.py` and `scripts/average_checkpoints.py` when v3
(or a later run) writes more than one snapshot (`best/` and `last/`).

The step is kept. The measured full-test gain is +0.13 exact points (57.7% →
57.8%), CER 8.73% → 8.60%, TER 6.66% → 6.58% — below the rough +0.5 to +2
estimate, but free and real.
