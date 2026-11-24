"""Tune decoding (length penalty, brace/grammar constraints, optional SWA) on a
fixed valid subset, then confirm the winner once on the full test set.

  python scripts/eval_decode.py \\
      --model results/kaggle_v2/run/best \\
      --valid data/raw/val-*.parquet --test data/raw/test-*.parquet \\
      --out results/decode_v2 \\
      --valid-limit 1000 --seed 0 --beams 4 --length-penalties 0.0 0.6 0.8 1.0

Encoder hidden states for the valid subset are cached under --out/cache so a
restart skips the expensive part. Each setting writes predictions and updates
metrics.json as it goes. The official v2 beam-4 baseline is length_penalty=1.0
(the HuggingFace default already used by scripts/eval.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.average import discover_epoch_checkpoints, save_averaged  # noqa: E402
from hw.data import MathWritingParquet  # noqa: E402
from hw.decode import encode, generate  # noqa: E402
from hw.latex_tok import from_texo  # noqa: E402
from hw.metrics import scores  # noqa: E402
from hw.model import load  # noqa: E402


def parse_files(patterns):
    import glob
    out = []
    for p in patterns:
        out += sorted(glob.glob(p)) if any(c in p for c in "*?[]") else [p]
    assert out, f"no files for {patterns}"
    return out


def tag_of(lp, constrain, beams, ckpt="best"):
    body = f"beam{beams}_lp{lp:g}" + ("_brace" if constrain else "")
    return body if ckpt == "best" else f"{ckpt}_{body}"


def setting_key(s):
    return (round(float(s["length_penalty"]), 4), bool(s["constrain"]), s.get("ckpt", "best"))


def rank_tuple(m):
    """Higher is better: exact, then -TER, then -CER."""
    return (m.get("exact_match", -1), -m.get("token_error_rate", 9), -m.get("cer", 9))


@torch.no_grad()
def encode_dataset(model, ds, device, batch_size, cache_path: Path | None, max_n=None):
    n = len(ds) if max_n is None else min(max_n, len(ds))
    if cache_path is not None and cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        if int(blob["n"]) == n:
            print(f"loaded encoder cache {cache_path} n={n}", flush=True)
            return blob["hidden"]
        print(f"ignoring stale encoder cache n={blob['n']} want {n}", flush=True)
    # probe one batch to estimate a full cache; refuse to materialise more than ~2 GB
    px0 = torch.stack([ds[j]["pixel_values"] for j in range(0, min(batch_size, n))])
    h0 = encode(model, px0.to(device)).to("cpu")
    bytes_est = int(h0.nelement() / h0.size(0) * n * h0.element_size())
    if bytes_est > 2 * 1024**3:
        print(f"encoder cache would be {bytes_est/1e9:.1f} GB; streaming instead (no cache)", flush=True)
        return None
    hs, t0 = [h0], time.time()
    for i in range(len(h0), n, batch_size):
        px = torch.stack([ds[j]["pixel_values"] for j in range(i, min(i + batch_size, n))])
        h = encode(model, px.to(device)).to("cpu")
        hs.append(h)
        if (i // batch_size) % 10 == 0 or i + batch_size >= n:
            print(f"  encode {min(i + batch_size, n)}/{n} in {time.time() - t0:.1f}s", flush=True)
    hidden = torch.cat(hs, 0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"n": n, "hidden": hidden}, cache_path)
    print(f"encoded {n} in {time.time() - t0:.1f}s, cache={cache_path}", flush=True)
    return hidden


@torch.no_grad()
def eval_setting(model, tok, ds, hidden, args, device, tag, length_penalty, constrain, beams):
    """If hidden is None, encode each batch on the fly (use this for large sets:
    a 7,644-sample fp32 encoder cache is ~8 GB and will OOM on a 16 GB box)."""
    out = Path(args.out)
    pred_path = out / f"predictions_{tag}.csv"
    rows, t0 = [], time.time()
    n = len(ds)
    bs = args.eval_batch_size
    for i in range(0, n, bs):
        j = min(i + bs, n)
        if hidden is None:
            px = torch.stack([ds[k]["pixel_values"] for k in range(i, j)])
            enc = encode(model, px.to(device))
        else:
            enc = hidden[i:j]
        raw = generate(model, tok, encoder_hidden_states=enc, device=device,
                       max_length=args.max_len, num_beams=beams,
                       length_penalty=length_penalty, constrain=constrain)
        for k, r in enumerate(raw):
            idx = i + k
            rows.append({"idx": idx, "sample_id": str(ds.sample_id[idx]), "gt": str(ds.latex[idx]),
                         "pred": from_texo(r), "pred_tokens": r})
        if (i // bs) % 5 == 0 or j >= n:
            print(f"  [{tag}] {j}/{n} in {time.time() - t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    m = scores(list(df["pred"]), list(df["gt"]))
    m.update({"beams": beams, "length_penalty": length_penalty, "constrain": constrain,
              "seconds": round(time.time() - t0, 1), "tag": tag})
    df.to_csv(pred_path, index=False)
    print(f"[{tag}] {m}", flush=True)
    return m, df


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def dump_metrics(path: Path, metrics: dict):
    path.write_text(json.dumps(metrics, indent=2))


def pick_winner(valid_metrics: dict):
    cands = [(k, v) for k, v in valid_metrics.items() if isinstance(v, dict) and "exact_match" in v]
    if not cands:
        return None
    return max(cands, key=lambda kv: rank_tuple(kv[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/kaggle_v2/run/best")
    ap.add_argument("--valid", nargs="+", default=["data/raw/val-*.parquet"])
    ap.add_argument("--test", nargs="+", default=["data/raw/test-*.parquet"])
    ap.add_argument("--out", default="results/decode_v2")
    ap.add_argument("--valid-limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--length-penalties", type=float, nargs="+", default=[0.0, 0.6, 0.8, 1.0])
    ap.add_argument("--brace", choices=["off", "on", "both"], default="both",
                    help="grammar-constrained decoding: off, on, or both (tune)")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--encode-batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--time-budget-h", type=float, default=1.75,
                    help="stop starting new valid settings after this many hours; always try the test confirm if a winner exists")
    ap.add_argument("--discover-ckpts", nargs="*", default=["results/kaggle_v2/run", "kaggle/ckpt_dataset"],
                    help="roots to search for extra epoch snapshots to average")
    ap.add_argument("--skip-average", action="store_true")
    ap.add_argument("--test-confirm", action="store_true", default=True)
    ap.add_argument("--no-test-confirm", action="store_false", dest="test_confirm")
    ap.add_argument("--limit-test", type=int, help="debug: cap the confirmation test set")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    budget_s = args.time_budget_h * 3600
    metrics = load_metrics(out / "metrics.json")
    metrics.setdefault("args", {k: (v if not isinstance(v, Path) else str(v))
                                for k, v in vars(args).items()})
    metrics["device"] = str(device)

    valid = MathWritingParquet(parse_files(args.valid), limit=args.valid_limit, seed=args.seed)
    print("device", device, "valid", len(valid), "model", args.model, flush=True)

    # --- optional SWA if 2+ snapshots of the same run exist ---
    avg_dir = None
    disc = {}
    if not args.skip_average:
        disc = discover_epoch_checkpoints(args.discover_ckpts)
        metrics["discovered_checkpoints"] = {k: [str(p) for p in v] for k, v in disc.items()}
        if disc:
            run, snaps = next(iter(disc.items()))
            avg_dir = out / "averaged"
            if not (avg_dir / "model.safetensors").exists() and not (avg_dir / "pytorch_model.bin").exists():
                save_averaged(snaps, avg_dir)
            metrics["averaged_from"] = [str(p) for p in snaps]
            print("will also eval averaged ckpt from", snaps, flush=True)
        else:
            metrics["averaged_from"] = []
            metrics["average_skipped"] = "no run with 2+ checkpoints (v2 only saved best; last was never written)"
            print(metrics["average_skipped"], flush=True)
    dump_metrics(out / "metrics.json", metrics)

    model, tok = load(args.model)
    model.to(device).eval()
    hidden = encode_dataset(model, valid, device, args.encode_batch_size, out / "cache" / "valid_encoder.pt")

    brace_opts = {"off": [False], "on": [True], "both": [False, True]}[args.brace]
    # length_penalty=1.0, constrain=False is the official baseline (HF default)
    settings = []
    for lp in args.length_penalties:
        for br in brace_opts:
            settings.append({"length_penalty": lp, "constrain": br, "ckpt": "best"})
    if avg_dir is not None:
        # only the default decode on the averaged weights, plus the later winner if needed
        settings.append({"length_penalty": 1.0, "constrain": False, "ckpt": "averaged"})

    valid_m = metrics.setdefault("valid", {})
    # skip already-finished tags (resume)
    def remaining_time():
        return budget_s - (time.time() - t_start)

    for s in settings:
        tag = tag_of(s["length_penalty"], s["constrain"], args.beams, s["ckpt"])
        if tag in valid_m and "exact_match" in valid_m[tag]:
            print(f"skip {tag}, already in metrics.json", flush=True)
            continue
        if remaining_time() < 60:
            print(f"time budget exhausted, skipping {tag}", flush=True)
            metrics.setdefault("skipped", []).append(tag)
            dump_metrics(out / "metrics.json", metrics)
            continue
        if s["ckpt"] == "averaged":
            del model
            model, tok = load(avg_dir)
            model.to(device).eval()
            # encoder of an averaged ckpt is different; re-encode
            hidden_avg = encode_dataset(model, valid, device, args.encode_batch_size,
                                        out / "cache" / "valid_encoder_averaged.pt")
            m, _ = eval_setting(model, tok, valid, hidden_avg, args, device, tag,
                                s["length_penalty"], s["constrain"], args.beams)
            # reload best for any later work
            del model
            model, tok = load(args.model)
            model.to(device).eval()
        else:
            m, _ = eval_setting(model, tok, valid, hidden, args, device, tag,
                                s["length_penalty"], s["constrain"], args.beams)
        m["ckpt"] = s["ckpt"]
        valid_m[tag] = m
        dump_metrics(out / "metrics.json", metrics)

    winner = pick_winner(valid_m)
    if winner is None:
        dump_metrics(out / "metrics.json", metrics)
        print("no valid results; stop")
        return
    wtag, wm = winner
    metrics["winner_valid"] = {"tag": wtag, **wm}
    print("winner on valid:", wtag, wm, flush=True)
    dump_metrics(out / "metrics.json", metrics)

    # --- confirm winner on the full test set (skip if it is the published baseline) ---
    baseline_tag = tag_of(1.0, False, args.beams, "best")
    if args.test_confirm:
        test = MathWritingParquet(parse_files(args.test), limit=args.limit_test, seed=args.seed)
        print("test", len(test), flush=True)
        if wtag == baseline_tag and args.limit_test is None:
            # official published numbers; do not spend another full pass
            metrics["test"] = {
                baseline_tag: {
                    "n": 7644, "cer": 0.08728872308697452, "token_error_rate": 0.06657569772175564,
                    "exact_match": 0.576792255363684, "beams": args.beams, "length_penalty": 1.0,
                    "constrain": False, "source": "results/kaggle_v2_eval/eval/metrics.json (not re-run)",
                    "tag": baseline_tag,
                }
            }
            metrics["test_note"] = "winner is the published v2 beam-4 baseline; full test was not re-run"
            print(metrics["test_note"], flush=True)
        else:
            ckpt = wm.get("ckpt", "best")
            if ckpt == "averaged":
                del model
                model, tok = load(avg_dir)
                model.to(device).eval()
            # Stream encode+decode. Do not materialise a full-test encoder cache
            # (7644 * 144 * 2048 * 4 B ≈ 8.4 GB).
            tm, _ = eval_setting(model, tok, test, None, args, device, f"test_{wtag}",
                                 wm["length_penalty"], wm["constrain"], args.beams)
            metrics.setdefault("test", {})[wtag] = tm
            # also record the published baseline next to it for the table
            metrics["test"][baseline_tag] = {
                "n": 7644, "cer": 0.08728872308697452, "token_error_rate": 0.06657569772175564,
                "exact_match": 0.576792255363684, "beams": 4, "length_penalty": 1.0,
                "constrain": False, "source": "results/kaggle_v2_eval/eval/metrics.json",
                "tag": baseline_tag,
            }
        dump_metrics(out / "metrics.json", metrics)

    write_summary(out, metrics)
    print(json.dumps({k: metrics[k] for k in metrics if k != "args"}, indent=2))


def write_summary(out: Path, metrics: dict):
    lines = [
        "# Decode tuning (roadmap step 2)",
        "",
        "Tune on a fixed valid subset (limit 1000, seed 0), then confirm the winner",
        "on the full MathWriting test set (7,644) with `hw.normalize` scoring.",
        "Baseline is v2 beam 4 with HuggingFace `length_penalty=1.0` (already the",
        "default in `scripts/eval.py`): 57.7% exact, CER 8.73%, TER 6.66%.",
        "",
        "## Valid subset",
        "",
        "| Setting | Exact | CER | TER | Seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    valid = metrics.get("valid", {})
    for tag, m in sorted(valid.items(), key=lambda kv: -rank_tuple(kv[1])[0] if "exact_match" in kv[1] else 0):
        if "exact_match" not in m:
            continue
        lines.append(
            f"| `{tag}` | {100*m['exact_match']:.2f}% | {100*m['cer']:.2f}% | "
            f"{100*m['token_error_rate']:.2f}% | {m.get('seconds', '')} |"
        )
    w = metrics.get("winner_valid")
    if w:
        lines += ["", f"Winner on valid: `{w['tag']}` "
                  f"({100*w['exact_match']:.2f}% exact, CER {100*w['cer']:.2f}%, "
                  f"TER {100*w['token_error_rate']:.2f}%)."]
    if metrics.get("average_skipped"):
        lines += ["", "Checkpoint averaging: " + metrics["average_skipped"]]
    elif metrics.get("averaged_from"):
        lines += ["", "Averaged checkpoints: " + ", ".join(metrics["averaged_from"])]
    test = metrics.get("test", {})
    if test:
        lines += ["", "## Full test (7,644)", "",
                  "| Setting | Exact | CER | TER |",
                  "|---|---:|---:|---:|"]
        for tag, m in test.items():
            if "exact_match" not in m:
                continue
            lines.append(f"| `{tag}` | {100*m['exact_match']:.2f}% | {100*m['cer']:.2f}% | "
                         f"{100*m['token_error_rate']:.2f}% |")
        lines.append("")
        if metrics.get("test_note"):
            lines.append(metrics["test_note"])
    rec = metrics.get("recommendation")
    if rec:
        lines += ["", "## Recommendation", "", rec]
    # fill recommendation if missing
    if not rec and w:
        base = (test.get(tag_of(1.0, False, w.get("beams", 4), "best"))
                or {"exact_match": 0.576792255363684, "cer": 0.08728872308697452,
                    "token_error_rate": 0.06657569772175564})
        wt = test.get(w["tag"], w)
        gain = 100 * (wt["exact_match"] - base["exact_match"])
        if w["tag"] == tag_of(1.0, False, w.get("beams", 4), "best") or gain <= 0.05:
            rec = ("Keep the current default: beam 4 with length_penalty=1.0 and no "
                   "grammar constraint. Nothing on the valid subset beat the published "
                   "baseline enough to change the decoder.")
        else:
            rec = (f"Use `{w['tag']}` at decode time (length_penalty={w['length_penalty']}"
                   f"{', grammar-constrained' if w['constrain'] else ''}). "
                   f"Full-test exact {100*wt['exact_match']:.2f}% vs baseline "
                   f"{100*base['exact_match']:.2f}% ({gain:+.2f} points).")
        metrics["recommendation"] = rec
        lines += ["", "## Recommendation", "", rec]
        dump_metrics(out / "metrics.json", metrics)
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("wrote", out / "SUMMARY.md", flush=True)


if __name__ == "__main__":
    main()
