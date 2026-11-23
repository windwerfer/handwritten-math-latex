"""Fine-tune Texo (alephpi/FormulaNet, 20M) on MathWriting and evaluate.

Data dir must contain train*.parquet, valid*.parquet (or val*.parquet) and test*.parquet,
either our subset format or HF deepcopy/MathWriting-human parquet shards.

  python scripts/train.py --data data/subset --model models/texo --out runs/smoke \\
      --train-limit 50 --test-limit 20 --max-steps 10 --batch-size 4

Labels/preds are canonicalised with hw.normalize (training targets and scoring).
Writes to --out: metrics.json, predictions_*.csv, samples/*.png, train_log.csv,
best/ (best valid-CER checkpoint) and last/ (unless --no-save).
"""
import argparse
import glob
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.data import MathWritingParquet, make_collate  # noqa: E402
from hw.latex_tok import from_texo, to_texo  # noqa: E402
from hw.metrics import scores  # noqa: E402
from hw.model import load, loss_fn, predict  # noqa: E402


def files(d, *prefixes):
    out = []
    for p in prefixes:
        out += sorted(glob.glob(str(Path(d) / f"{p}*.parquet")))
    assert out, f"no {prefixes} parquet in {d}"
    return out


def evaluate(model, tok, ds, args, device, tag, beams=1, save=True):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                                     num_workers=args.workers, collate_fn=make_collate(tok))
    rows, t0 = [], time.time()
    for b in dl:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            raw = predict(model, tok, b["pixel_values"], device, max_length=args.max_len,
                          num_beams=beams)
        for i, r, gt in zip(b["idx"], raw, b["latex"]):
            rows.append({"idx": i, "sample_id": ds.df.sample_id[i], "gt": gt,
                         "pred": from_texo(r), "pred_tokens": r})
    df = pd.DataFrame(rows)
    m = scores(list(df["pred"]), list(df["gt"]))
    m["beams"] = beams
    m["seconds"] = round(time.time() - t0, 1)
    if save:
        df.to_csv(Path(args.out) / f"predictions_{tag}.csv", index=False)
    print(f"[{tag}] {m}", flush=True)
    return m, df


def save_samples(ds, dfs: dict, out, n=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    first = next(iter(dfs.values()))
    for k in range(min(n, len(first))):
        r = first.iloc[k]
        fig = plt.figure(figsize=(8, 3.4))
        ax = fig.add_axes([0, 0.4, 1, 0.6]); ax.imshow(ds.image(int(r["idx"]))); ax.axis("off")
        lines = [f"{'GT':<11} {from_texo(to_texo(r['gt']))}"]
        for name, df in dfs.items():
            lines.append(f"{name:<11} {df.iloc[k]['pred']}")
        fig.text(0.01, 0.02, "\n".join(lines), family="monospace", fontsize=8, va="bottom")
        fig.savefig(out / f"{k:02d}_{r['sample_id']}.png", dpi=100); plt.close(fig)


def bare(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/subset")
    ap.add_argument("--model", default="models/texo")
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--train-limit", type=int)
    ap.add_argument("--valid-limit", type=int, help="random valid subset used for per-epoch selection")
    ap.add_argument("--test-limit", type=int)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (0 = off)")
    ap.add_argument("--time-budget-h", type=float, default=0,
                    help="cap training wall time; epochs are reduced to fit (0 = off)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--beams", type=int, nargs="*", default=[1], help="beam sizes for final test eval")
    ap.add_argument("--final-full-valid", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-zeroshot", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--data-parallel", action="store_true", help="nn.DataParallel over all GPUs")
    ap.add_argument("--bn-train", action="store_true",
                    help="update BatchNorm running stats (default: keep pretrained stats frozen)")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import transformers
    print("device", device, torch.cuda.device_count(), "torch", torch.__version__,
          "transformers", transformers.__version__, flush=True)

    model, tok = load(args.model)
    model.to(device)
    print("params (M)", sum(p.numel() for p in model.parameters()) / 1e6, flush=True)
    d = Path(args.data)
    train = MathWritingParquet(files(d, "train"), train=True, limit=args.train_limit)
    valid = MathWritingParquet(files(d, "valid", "val-"), limit=args.valid_limit)
    test = MathWritingParquet(files(d, "test"), limit=args.test_limit)
    print("sizes", len(train), len(valid), len(test), flush=True)

    metrics = {"args": vars(args)}
    preds = {}
    if not args.skip_zeroshot:
        metrics["test_zeroshot"], preds["zero-shot"] = evaluate(model, tok, test, args, device, "zeroshot")

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    net = torch.nn.DataParallel(model) if args.data_parallel and torch.cuda.device_count() > 1 else model
    dl = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                     num_workers=args.workers, collate_fn=make_collate(tok, args.max_len),
                                     persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    plan = {"epochs": args.epochs, "total": args.max_steps or args.epochs * len(dl)}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, plan["total"]) / plan["total"])))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    log, step, t0, best = [], 0, time.time(), None
    budget_s = args.time_budget_h * 3600
    ep = 0
    while ep < plan["epochs"]:
        model.train()
        if not args.bn_train:  # small fine-tuning batches give bad BN statistics
            for mod in model.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        for b in dl:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(net, b, device)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 50 == 0 or step == 1:
                log.append({"step": step, "epoch": ep, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                            "elapsed_s": round(time.time() - t0, 1)})
                print(log[-1], flush=True)
            if step == 300 and budget_s and not args.max_steps:
                # estimate epoch time from the first 300 steps (+ ~3% for eval) and fit the budget
                ep_s = (time.time() - t0) / 300 * len(dl) * 1.03
                fit = max(1, int(budget_s // ep_s))
                if fit < plan["epochs"]:
                    plan["epochs"] = fit; plan["total"] = fit * len(dl)
                print(f"[plan] ~{ep_s/60:.1f} min/epoch, {len(dl)} steps/epoch, budget {args.time_budget_h}h "
                      f"-> {plan['epochs']} epochs ({plan['total']} steps)", flush=True)
                metrics["plan"] = {"min_per_epoch": round(ep_s / 60, 1), "steps_per_epoch": len(dl), **plan}
            if args.max_steps and step >= args.max_steps:
                break
        m, _ = evaluate(model, tok, valid, args, device, f"valid_ep{ep}", save=False)
        metrics.setdefault("valid", []).append({"epoch": ep, "step": step,
                                                "elapsed_min": round((time.time() - t0) / 60, 1), **m})
        if not args.no_save and (best is None or m["cer"] < best["cer"]):
            best = {"epoch": ep, **m}
            model.save_pretrained(out / "best"); tok.save_pretrained(out / "best")
        metrics["best_valid"] = best
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        ep += 1
        if args.max_steps and step >= args.max_steps:
            break
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["epochs_done"] = ep
    if not args.no_save:
        model.save_pretrained(out / "last"); tok.save_pretrained(out / "last")
        if best is not None:  # evaluate the best checkpoint
            model, tok = load(out / "best"); model.to(device)
    for nb in args.beams:
        tag = f"finetuned_beam{nb}"
        metrics[f"test_{tag}"], preds[f"ft beam{nb}"] = evaluate(model, tok, test, args, device, tag, beams=nb)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if args.final_full_valid:
        fv = MathWritingParquet(files(d, "valid", "val-"))
        metrics["valid_full_beam1"], _ = evaluate(model, tok, fv, args, device, "valid_full", save=False)
    save_samples(test, preds, out / "samples")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()
