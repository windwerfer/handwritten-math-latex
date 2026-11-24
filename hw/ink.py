"""MathWriting InkML parsing, stroke-level augmentation and rendering.

Rendering mimics the HF mirror deepcopy/MathWriting-human used for valid/test (black-ish
strokes on white, roughly native ink scale with the longer side capped at 512 px, stroke
width ~1.5-4 px), so that ink-rendered training images match the evaluation domain.
"""
import re

import cv2
import numpy as np
from PIL import Image

_ANN = re.compile(r'<annotation type="([^"]+)">(.*?)</annotation>', re.S)
_TRACE = re.compile(r"<trace(?:\s[^>]*)?>(.*?)</trace>", re.S)


def _unescape(s):
    return s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")


def parse_inkml(text: str):
    """-> (annotations dict, list of float32 [N,2] strokes (x, y))."""
    ann = {k: _unescape(v) for k, v in _ANN.findall(text)}
    strokes = []
    for tr in _TRACE.findall(text):
        tr = tr.strip().rstrip(",")
        if not tr:
            continue
        ncol = len(tr.split(",", 1)[0].split())
        a = np.array(tr.replace(",", " ").split(), dtype=np.float32).reshape(-1, ncol)[:, :2]
        strokes.append(np.ascontiguousarray(a))
    return ann, strokes


def label_of(ann: dict) -> str:
    return ann.get("normalizedLabel") or ann.get("label", "")


def augment_strokes(strokes, rng: np.random.Generator, strength=1.0):
    """Stroke-level augmentation (train only):
    small per-formula affine (rotation, shear, anisotropic scale), per-stroke offset,
    per-point jitter, and random light smoothing. Scales are relative to the formula height."""
    allp = np.concatenate(strokes)
    lo, hi = allp.min(0), allp.max(0)
    h = max(hi[1] - lo[1], 1e-3)
    w = max(hi[0] - lo[0], 1e-3)
    unit = min(h, w * 0.5) if w < h else h  # ~ character height for one-line formulas
    c = (lo + hi) / 2
    s = strength
    ang = np.deg2rad(rng.uniform(-3, 3) * s)
    sh = rng.uniform(-0.15, 0.15) * s
    sx, sy = 1 + rng.uniform(-0.1, 0.1) * s, 1 + rng.uniform(-0.1, 0.1) * s
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    A = R @ np.array([[1, sh], [0, 1]], np.float32) @ np.diag([sx, sy]).astype(np.float32)
    k = rng.choice([1, 1, 3, 5])  # smoothing window (1 = none)
    out = []
    for st in strokes:
        p = (st - c) @ A.T + c
        p = p + rng.normal(0, 0.012 * unit * s, size=2)          # per-stroke offset
        p = p + rng.normal(0, 0.004 * unit * s, size=p.shape)    # point jitter
        if k > 1 and len(p) >= k:
            ker = np.ones(k, np.float32) / k
            pad = np.pad(p, ((k // 2, k // 2), (0, 0)), mode="edge")
            p = np.stack([np.convolve(pad[:, d], ker, mode="valid") for d in range(2)], 1)
        out.append(p.astype(np.float32))
    return out


def render(strokes, rng: np.random.Generator | None = None, side=None, width=None, margin=4,
           hrange=(120, 320), wdiv=28):
    """Render strokes to an RGB PIL image (white background, dark strokes).
    side: longer image side in px. The deepcopy renders have the longer side anywhere between
    ~100 and 512 px with 2-4 px strokes, i.e. the stroke width *relative to the formula*
    varies a lot after the model's fit-to-384 resize; by default we sample side log-uniformly."""
    rng = rng or np.random.default_rng(0)
    allp = np.concatenate(strokes)
    lo, hi = allp.min(0), allp.max(0)
    ext = np.maximum(hi - lo, 1.0)
    if side is None:
        # formula height ~120-320 px, longer side capped at 512 px (as in the deepcopy renders);
        # calibrated so the ink fraction after preprocessing matches deepcopy (0.057 vs 0.058)
        hpx = float(np.exp(rng.uniform(np.log(hrange[0]), np.log(hrange[1]))))
        side = min(512.0, hpx * ext.max() / ext[1])
    scale = (side - 2 * margin) / ext.max()
    W, H = (np.ceil(ext * scale) + 2 * margin).astype(int)
    if width is None:
        width = int(rng.choice([1, 2, 2, 2, 3]))  # calibrated: ink fraction ~ deepcopy renders
        width = max(1, min(width, int(round(ext[1] * scale / wdiv))))  # thin strokes for small glyphs
    S = 4  # sub-pixel precision for cv2 antialiasing
    img = np.full((H, W), 255, np.uint8)
    ink = int(rng.integers(0, 60))
    for st in strokes:
        q = ((st - lo) * scale + margin) * (1 << S)
        q = np.round(q).astype(np.int32).reshape(-1, 1, 2)
        if len(q) == 1:
            cv2.circle(img, tuple(q[0, 0]), max(1, width // 2) << S, ink, -1, cv2.LINE_AA, S)
        else:
            cv2.polylines(img, [q], False, ink, width, cv2.LINE_AA, S)
    rgb = np.repeat(img[:, :, None], 3, 2)
    if rng.random() < 0.5:  # subtle colour perturbation like the deepcopy renders
        rgb = np.clip(rgb.astype(np.int16) + rng.integers(-8, 9, size=3), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)
