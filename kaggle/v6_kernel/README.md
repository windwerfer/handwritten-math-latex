# v6: continue mixed training from v5-best

Same data mix as v5 (`--color-strokes --color-mix 0.70 --human-image-prob 0.40
--eval-plain-images`). Starts from `v5best`. ONNX export is skipped; the kernel
runs colour, greyscale InkML, and deepcopy evals (greedy + beam 4) plus a full
valid greedy pass at the end.

## Learning rate

v5 used a cosine schedule with peak `3e-5` over the (budget-fitted) 4-epoch
horizon, so the last logged LR was ~4e-12. A new `train.py` process does not
reload the optimiser, but reusing `3e-5` would jump straight back to the
original peak.

v6 therefore uses a **fresh, lower cosine**: peak **`1.5e-5`** (half of v5)
and **warmup 300 steps** (~2–3 min). After the smoke gate, `plan["total"]` is
fitted to the 5-epoch / time-cap horizon so the cosine covers this continue
session rather than decaying as if 10 epochs remained. This is the usual
"continue fine-tune at a lower peak" schedule: enough to keep improving
(valid was still rising at v5 epoch 3) without a large LR shock.

## Time cap

Fresh weekly quota is 30 GPU-h (session cap still 12 h). The run is sized to
**about 5 epochs** at v5's 95.4 min/epoch, not the full 12 h: training budget
`min(session leftover, 8.30 h)` plus a 0.35 h reserve for the final evals.
`--epochs 5` is the planned stop; the time cap is a safety net if an epoch
runs long.
