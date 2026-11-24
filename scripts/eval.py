"""Evaluate one or more checkpoints on a test set with the canonical (hw.normalize) scoring.

  python scripts/eval.py --test data/raw/test-*.parquet --out runs/eval \\
      --model zeroshot=models/texo:1 --model v1=path:1 --model v2=path:1,4

Each --model is NAME=PATH:BEAMS (comma-separated beam sizes). Writes metrics.json,
predictions_<name>_beam<k>.csv and samples/*.png comparing all models.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import evaluate, save_samples  # noqa: E402
from hw.data import MathWritingParquet  # noqa: E402
from hw.model import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--length-penalty", type=float, default=0.0,
                    help="HF beam length_penalty (alpha). 0.0 is recommended (see results/decode_v2/)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = MathWritingParquet(args.test, limit=args.limit)
    print("device", device, "test", len(test), flush=True)
    metrics, preds = {}, {}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        path, beams = rest.rsplit(":", 1)
        model, tok = load(path); model.to(device)
        for nb in [int(b) for b in beams.split(",")]:
            tag = f"{name}_beam{nb}"
            try:
                metrics[tag], preds[tag] = evaluate(model, tok, test, args, device, tag, beams=nb)
            except Exception as e:  # noqa: BLE001
                metrics[tag] = {"error": repr(e)}
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        del model; torch.cuda.empty_cache() if device.type == "cuda" else None
    save_samples(test, {k: v for k, v in preds.items()}, out / "samples", n=16)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
