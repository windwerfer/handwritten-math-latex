"""Stream the official MathWriting-2024 tgz and pack inks into on-disk numpy arrays.

  python scripts/prep_inks.py --tgz mathwriting-2024.tgz --out /tmp/inks --splits train synthetic

Per split writes (memmap-friendly, no Python objects at train time):
  {split}_points.npy  float32 [P,2]   all points
  {split}_soff.npy    int64   [S+1]   stroke start offsets into points
  {split}_ioff.npy    int64   [N+1]   ink start offsets into strokes
  {split}_meta.parquet  sample_id, latex (normalizedLabel), n_tokens (Texo tokens)
Inks whose label has > --max-tokens Texo tokens are skipped.
"""
import argparse
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.ink import label_of, parse_inkml  # noqa: E402
from hw.latex_tok import to_texo  # noqa: E402


class Packer:
    def __init__(self):
        self.pts, self.slen, self.ilen, self.sid, self.lab, self.ntok = [], [], [], [], [], []

    def add(self, sid, lab, ntok, strokes):
        self.pts += strokes
        self.slen += [len(s) for s in strokes]
        self.ilen.append(len(strokes))
        self.sid.append(sid); self.lab.append(lab); self.ntok.append(ntok)

    def save(self, out: Path, split: str):
        np.save(out / f"{split}_points.npy", np.concatenate(self.pts).astype(np.float32))
        np.save(out / f"{split}_soff.npy", np.concatenate([[0], np.cumsum(self.slen)]).astype(np.int64))
        np.save(out / f"{split}_ioff.npy", np.concatenate([[0], np.cumsum(self.ilen)]).astype(np.int64))
        pd.DataFrame({"sample_id": self.sid, "latex": self.lab, "n_tokens": self.ntok}).to_parquet(
            out / f"{split}_meta.parquet", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "synthetic"])
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0, help="max inks per split (0 = all)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    packs = {s: Packer() for s in a.splits}
    skipped = {s: 0 for s in a.splits}
    t0 = time.time(); n = 0
    with tarfile.open(a.tgz, "r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".inkml"):
                continue
            if a.limit and all(len(p.sid) >= a.limit for p in packs.values()):
                break
            split = m.name.split("/")[-2]
            if split not in packs or (a.limit and len(packs[split].sid) >= a.limit):
                continue
            ann, strokes = parse_inkml(tf.extractfile(m).read().decode("utf-8"))
            lab = label_of(ann)
            ntok = len(to_texo(lab).split())
            if not strokes or ntok > a.max_tokens:
                skipped[split] += 1
                continue
            packs[split].add(os.path.basename(m.name)[:-6], lab, ntok, strokes)
            n += 1
            if n % 50000 == 0:
                print(f"{n} inks, {time.time() - t0:.0f}s", {k: len(v.sid) for k, v in packs.items()}, flush=True)
    for s, p in packs.items():
        p.save(out, s)
        print(s, "inks", len(p.sid), "skipped", skipped[s], flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
