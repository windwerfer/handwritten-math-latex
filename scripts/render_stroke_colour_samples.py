"""Render a few official inks in stroke-colour and greyscale for a visual check.

  python scripts/render_stroke_colour_samples.py \\
      --ink-dir data/excerpt/packed --out results/v4_stroke_colour_samples
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.data import InkArrays  # noqa: E402
from hw.ink import render, render_eval  # noqa: E402


def split_channels(img: Image.Image) -> Image.Image:
    """Side-by-side: RGB | R | G | B (grey)."""
    a = np.asarray(img)
    parts = [img]
    for c, name in enumerate("RGB"):
        ch = Image.fromarray(a[:, :, c])
        parts.append(ImageOps.colorize(ch, black="black", white="white") if False else ch.convert("RGB"))
        # show channel as a monochrome RGB image
        mono = np.repeat(a[:, :, c:c+1], 3, 2)
        parts[-1] = Image.fromarray(mono)
    # pad to common height, concat horizontally with a 4 px white gutter
    h = max(p.height for p in parts)
    out = []
    for p in parts:
        if p.height != h:
            p = ImageOps.pad(p, (p.width, h), color=(255, 255, 255), centering=(0.5, 0.5))
        out.append(p)
        out.append(Image.new("RGB", (4, h), (240, 240, 240)))
    w = sum(p.width for p in out)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for p in out:
        canvas.paste(p, (x, 0)); x += p.width
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ink-dir", default="data/excerpt/packed")
    ap.add_argument("--out", default="results/v4_stroke_colour_samples")
    ap.add_argument("--per-split", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "synthetic", "valid"):
        meta = Path(args.ink_dir, f"{split}_meta.parquet")
        if not meta.exists():
            print("skip", split); continue
        inks = InkArrays(args.ink_dir, split)
        for i in range(min(args.per_split, len(inks))):
            st = inks.strokes(i)
            sid = str(inks.sample_id[i])
            colour = render_eval(st, color=True)
            grey = render_eval(st, color=False)
            colour.save(out / f"{split}_{sid}_colour.png")
            grey.save(out / f"{split}_{sid}_grey.png")
            split_channels(colour).save(out / f"{split}_{sid}_channels.png")
            print(split, sid, colour.size, "latex", str(inks.latex[i])[:80])
    print("wrote", out)


if __name__ == "__main__":
    main()
