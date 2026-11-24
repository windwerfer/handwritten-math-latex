# Decode tuning on v3 best (roadmap step 2, re-run)

Same protocol as `results/decode_v2/`: tune on a fixed valid subset (1,000
formulas, seed 0), then confirm the winner on the official MathWriting test
set (all 7,644 human inks) with `hw.normalize` scoring. CPU, PyTorch.

```bash
python scripts/eval_decode.py --model results/kaggle_v3/run/best \
    --valid 'data/raw/val-*.parquet' --test 'data/raw/test-*.parquet' \
    --out results/decode_v3 --valid-limit 1000 --seed 0 --beams 4 \
    --length-penalties 0.0 1.0 --brace off --discover-ckpts results/kaggle_v3/run
```

v3 saved two snapshots, `best/` (epoch 4) and `last/` (epoch 5), so
checkpoint averaging could be tried this time (`hw/average.py`, uniform mean
of the two state dicts).

## Valid subset (1,000, seed 0)

| Setting | Exact | CER | TER | Seconds |
|---|---:|---:|---:|---:|
| `beam4_lp0` | **71.70%** | **5.69%** | **4.99%** | 46.2 |
| `beam4_lp1` | 71.70% | 5.78% | 5.00% | 43.2 |
| averaged (best + last), `beam4_lp1` | 71.60% | 5.79% | 5.02% | 78.3 |

Winner on valid: `beam4_lp0` (tie on exact match, lower CER/TER).
Averaging best (epoch 4) + last (epoch 5) did not help: the two epochs are
close (valid CER 5.29% vs 5.33%), so their mean is not a better point.

## Full test (7,644)

| Setting | Exact | CER | TER | Exact count |
|---|---:|---:|---:|---:|
| `beam4_lp0` (this run) | 62.55% | **7.39%** | **5.49%** | 4781 |
| `beam4_lp1` (Kaggle v3 run, `results/kaggle_v3/run/metrics.json`) | **62.56%** | 7.43% | 5.56% | 4782 |
| greedy (Kaggle v3 run) | 61.80% | 7.79% | 5.84% | – |
| v2 best `beam4_lp0` (previous best) | 57.81% | 8.60% | 6.58% | 4419 |

On v3, `lp0` and `lp1` differ on 95 of 7,644 predictions: `lp0` is exact on
15 that `lp1` misses and `lp1` on 16 that `lp0` misses (net −1). The v2
over-structuring pattern still shows up in the `lp1` losses (`1^{1/2}` →
`1^{\frac{1}{2}}`, `De` → `D_{e}`, `\Omega m` → `\Omega_{m}`, `exp` →
`ex_{7}`), but now `lp0` loses a similar number by dropping structure
(`P_{a1}` → `Pa_{1}`, `\mu(dx)` → `dx`, a matrix entry dropped).

## Recommendation

Keep `length_penalty=0.0` as the default: exact match is a tie on v3, while
CER and TER are slightly lower. Brace-constrained decoding was not re-run
(no-op on v2). Checkpoint averaging of best + last gives no gain; a longer
run with more snapshots (or EMA) would be needed to test SWA properly.
