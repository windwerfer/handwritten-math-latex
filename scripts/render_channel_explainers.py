"""README explainer images: one stroke-colour channel per image, tinted red / green / blue.

Same ink, same features as the model's colour render (hw.ink.stroke_features), but each
image shows only one feature as shades of its own hue, plus a gradient legend.

  python scripts/render_channel_explainers.py --ink-dir data/excerpt/packed --split train --index 0
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.data import InkArrays  # noqa: E402
from hw.ink import stroke_features  # noqa: E402

HUES = {  # value 0 -> dark, value 1 -> bright, like the parser (ink + 140 * value)
    "red": ((70, 0, 0), (255, 120, 120)),
    "green": ((0, 70, 0), (120, 235, 120)),
    "blue": ((0, 0, 90), (0, 0, 90)),
}
TEXT = {
    "red": ("R = time", "first stroke", "last stroke"),
    "green": ("G = stroke direction", "", ""),
    "blue": ("B = ink (constant darkness)", "", ""),
}


def shade(hue, v):
    lo, hi = np.array(HUES[hue][0], float), np.array(HUES[hue][1], float)
    return tuple(int(x) for x in lo + (hi - lo) * float(np.clip(v, 0, 1)))


def font(sz):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if Path(f).exists():
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def draw(strokes, hue, hpx=170, width=4, margin=12):
    times, dirs = stroke_features(strokes)
    allp = np.concatenate(strokes); lo, hi = allp.min(0), allp.max(0)
    ext = np.maximum(hi - lo, 1.0); scale = hpx / ext[1]
    W, H = (np.ceil(ext * scale) + 2 * margin).astype(int)
    img = np.full((H, W, 3), 255, np.uint8); S = 4
    for st, ts, ds in zip(strokes, times, dirs):
        q = np.round(((st - lo) * scale + margin) * (1 << S)).astype(np.int32)
        val = ts if hue == "red" else ds if hue == "green" else np.zeros(len(st))
        for i in range(len(q)):
            bgr = shade(hue, val[i])  # array stays RGB (no cv2 colour conversion)
            if i == 0:
                cv2.circle(img, (int(q[0, 0]), int(q[0, 1])), (width // 2) << S, bgr, -1, cv2.LINE_AA, S)
            else:
                cv2.line(img, (int(q[i - 1, 0]), int(q[i - 1, 1])), (int(q[i, 0]), int(q[i, 1])),
                         bgr, width, cv2.LINE_AA, S)
    ink = Image.fromarray(img)
    title, left, right = TEXT[hue]
    pad, bar_h = 14, 14
    out = Image.new("RGB", (max(W, 520), H + 90), "white"); d = ImageDraw.Draw(out)
    d.text((pad, 8), title, fill=shade(hue, 0.0), font=font(22))
    out.paste(ink, (0, 38))
    y = 38 + H + 6; bw = out.width - 2 * pad
    for x in range(bw):
        d.line([(pad + x, y), (pad + x, y + bar_h)], fill=shade(hue, x / max(bw - 1, 1)))
    if hue == "green":  # u = (atan2(dy, dx) + pi) / 2pi, image y points down
        for frac, lab in ((0, "← left"), (0.25, "↑ up"), (0.5, "→ right"), (0.75, "↓ down"), (1, "← left")):
            tw = d.textlength(lab, font=font(15))
            x = min(max(pad + frac * bw - tw / 2, pad), pad + bw - tw)
            d.text((x, y + bar_h + 4), lab, fill=(80, 80, 80), font=font(15))
    if left:
        d.text((pad, y + bar_h + 4), left, fill=(80, 80, 80), font=font(15))
        tw = d.textlength(right, font=font(15))
        d.text((pad + bw - tw, y + bar_h + 4), right, fill=(80, 80, 80), font=font(15))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ink-dir", default="data/excerpt/packed")
    ap.add_argument("--split", default="train")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="docs/img")
    a = ap.parse_args()
    inks = InkArrays(a.ink_dir, a.split); st = inks.strokes(a.index)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("sample", inks.sample_id[a.index], inks.latex[a.index])
    for hue, name in (("red", "time"), ("green", "direction"), ("blue", "ink")):
        p = out / f"channel_{hue}_{name}.png"; draw(st, hue).save(p); print("wrote", p)


if __name__ == "__main__":
    main()
