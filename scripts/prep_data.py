"""Build a small MathWriting subset from the HF mirror deepcopy/MathWriting-human
(official MathWriting-2024 human train/valid/test splits, pre-rendered to images).

Usage: python scripts/prep_data.py --train 5000 --valid 500 --test 500
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.latex_tok import tokenize  # noqa: E402

RAW = Path("data/raw")
OUT = Path("data/subset")


def load(pattern):
    files = sorted(glob.glob(str(RAW / pattern)))
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    df["image_bytes"] = df.image.map(lambda d: d["bytes"])
    return df.drop(columns=["image"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--valid", type=int, default=500)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="models/texo")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    vocab = set(tok.get_vocab())
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, pat, n in [("train", "train-*.parquet", a.train), ("valid", "val-*.parquet", a.valid),
                          ("test", "test-*.parquet", a.test)]:
        df = load(pat)
        full = len(df)
        oov = df.latex.map(lambda s: any(t not in vocab for t in tokenize(s)))
        sub = df.sample(n=min(n, full), random_state=a.seed).reset_index(drop=True)
        sub_oov = sub.latex.map(lambda s: any(t not in vocab for t in tokenize(s)))
        sub.to_parquet(OUT / f"{split}.parquet", index=False)
        oov_tokens = pd.Series([t for s in df.latex for t in tokenize(s) if t not in vocab]).value_counts()
        stats[split] = {"source_rows": full, "subset_rows": len(sub),
                        "source_frac_with_oov": float(oov.mean()),
                        "subset_frac_with_oov": float(sub_oov.mean()),
                        "top_oov_tokens": oov_tokens.head(15).to_dict()}
        print(split, stats[split])
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
