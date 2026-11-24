"""Average two or more checkpoints of the same run and write a loadable HF folder.

  python scripts/average_checkpoints.py --out models/swa \\
      --ckpt runs/v3/best --ckpt runs/v3/last

  python scripts/average_checkpoints.py --out models/swa --discover runs/v3 results/kaggle_v2/run

Exits 0 with a message (and writes no folder) if --discover finds fewer than
two snapshots; v2 only shipped `best`, so averaging is a no-op until a run
saves more than one epoch.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.average import discover_epoch_checkpoints, save_averaged  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", default=[], help="checkpoint directory (repeatable)")
    ap.add_argument("--discover", nargs="*", help="run directories (or parents) to search for best/last/epoch_*")
    args = ap.parse_args()
    paths = [Path(p) for p in args.ckpt]
    report = {}
    if args.discover is not None:
        found = discover_epoch_checkpoints(args.discover)
        report["discovered"] = {k: [str(p) for p in v] for k, v in found.items()}
        if not paths:
            if len(found) == 1:
                paths = next(iter(found.values()))
            elif len(found) > 1:
                print(json.dumps(report, indent=2))
                sys.exit("several runs found; pass one with --ckpt or a more specific --discover")
    report["paths"] = [str(p) for p in paths]
    if len(paths) < 2:
        report["averaged"] = False
        report["reason"] = "need at least two checkpoints of the same run"
        print(json.dumps(report, indent=2))
        return
    out = save_averaged(paths, args.out)
    report["averaged"] = True
    report["out"] = str(out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
