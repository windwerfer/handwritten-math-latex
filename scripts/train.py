"""Fine-tune Texo (alephpi/FormulaNet, 20M) on a MathWriting subset and evaluate.

  python scripts/train.py --data data/subset --model models/texo --out runs/smoke \
      --train-limit 50 --test-limit 20 --max-steps 10 --batch-size 4

Writes to --out: metrics.json, predictions_{zeroshot,finetuned}.csv, samples/*.png,
train_log.csv and (unless --no-save) the fine-tuned model in --out/model.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.data import MathWritingParquet, make_collate  # noqa: E402
from hw.latex_tok import from_texo  # noqa: E402
from hw.metrics import scores  # noqa: E402
from hw.model import load, loss_fn, predict  # noqa: E402


def evaluate(model, tok, ds, args, device, tag):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                                     num_workers=args.workers, collate_fn=make_collate(tok))
    rows, t0 = [], time.time()
    for b in dl:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            raw = predict(model, tok, b["pixel_values"], device, max_length=args.max_len,
                          num_beams=args.beams)
        for i, r, gt in zip(b["idx"], raw, b["latex"]):
            rows.append({"idx": i, "sample_id": ds.df.sample_id[i], "gt": gt,
                         "pred": from_texo(r), "pred_tokens": r})
    df = pd.DataFrame(rows)
    m = scores(list(df["pred"]), list(df["gt"]))
    m["seconds"] = round(time.time() - t0, 1)
    df.to_csv(Path(args.out) / f"predictions_{tag}.csv", index=False)
    print(f"[{tag}] {m}", flush=True)
    return m, df


def save_samples(ds, zs, ft, out, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    merged = zs.merge(ft[["idx", "pred"]], on="idx", suffixes=("_zeroshot", "_finetuned")) if ft is not None else zs.rename(columns={"pred": "pred_zeroshot"})
    for k, r in merged.head(n).iterrows():
        img = ds.image(r["idx"])
        fig = plt.figure(figsize=(8, 3.2))
        ax = fig.add_axes([0, 0.38, 1, 0.62]); ax.imshow(img); ax.axis("off")
        txt = f"GT:         {r['gt']}\nzero-shot: {r['pred_zeroshot']}"
        if "pred_finetuned" in r:
            txt += f"\nfine-tuned: {r['pred_finetuned']}"
        fig.text(0.01, 0.02, txt, family="monospace", fontsize=8, va="bottom")
        fig.savefig(out / f"{k:02d}_{r['sample_id']}.png", dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/subset")
    ap.add_argument("--model", default="models/texo")
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--train-limit", type=int)
    ap.add_argument("--valid-limit", type=int)
    ap.add_argument("--test-limit", type=int)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (0 = off)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-zeroshot", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bn-train", action="store_true",
                    help="update BatchNorm running stats (default: keep pretrained stats frozen)")
    ap.add_argument("--freeze-encoder", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", flush=True)

    model, tok = load(args.model)
    model.to(device)
    print("params (M)", sum(p.numel() for p in model.parameters()) / 1e6, flush=True)
    d = Path(args.data)
    train = MathWritingParquet(d / "train.parquet", train=True, limit=args.train_limit)
    valid = MathWritingParquet(d / "valid.parquet", limit=args.valid_limit)
    test = MathWritingParquet(d / "test.parquet", limit=args.test_limit)
    print("sizes", len(train), len(valid), len(test), flush=True)

    metrics = {"args": vars(args)}
    zs_df = None
    if not args.skip_zeroshot:
        metrics["test_zeroshot"], zs_df = evaluate(model, tok, test, args, device, "zeroshot")

    dl = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                     num_workers=args.workers, collate_fn=make_collate(tok, args.max_len))
    total = args.max_steps or args.epochs * len(dl)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    log, step, t0 = [], 0, time.time()
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    for ep in range(args.epochs):
        model.train()
        if not args.bn_train:  # small fine-tuning batches give bad BN statistics
            for mod in model.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        for b in dl:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model, b, device)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            log.append({"step": step, "epoch": ep, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                        "elapsed_s": round(time.time() - t0, 1)})
            if step % 10 == 0 or step == 1:
                print(log[-1], flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if len(valid):
            m, _ = evaluate(model, tok, valid, args, device, f"valid_ep{ep}")
            metrics.setdefault("valid", []).append({"epoch": ep, "step": step, **m})
        if args.max_steps and step >= args.max_steps:
            break
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["test_finetuned"], ft_df = evaluate(model, tok, test, args, device, "finetuned")
    if zs_df is not None:
        save_samples(test, zs_df, ft_df, out / "samples")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if not args.no_save:
        model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
    print(json.dumps({k: v for k, v in metrics.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()
