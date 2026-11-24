"""MathWriting InkML parsing, stroke-level augmentation and rendering.

Rendering mimics the HF mirror deepcopy/MathWriting-human used for valid/test (black-ish
strokes on white, roughly native ink scale with the longer side capped at 512 px, stroke
width ~1.5-4 px), so that ink-rendered training images match the evaluation domain.

color=True encodes stroke features in the three RGB channels (Texo's HGNetv2 stem is
already 3-channel; hw.data.preprocess must be called with color=True so they are not
flattened to greyscale):
  R = normalised arc-length time along the whole ink (0 at the first point, 1 at the last)
  G = stroke direction (atan2 of the segment, mapped to [0, 1])
  B = ink darkness
All three stay in a dark range on the stroke so a greyscale-pretrained stem still sees
ink on paper; fine-tuning teaches it to use the channel differences.
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


def stroke_features(strokes):
    """Per-stroke arrays: t = normalised arc length in [0, 1] along the whole ink;
    d = segment direction in [0, 1] ((atan2 + pi) / 2pi). Singleton strokes get
    the mid-stroke time and d = 0.5."""
    seglen = []
    for st in strokes:
        if len(st) == 1:
            seglen.append(np.zeros(1, np.float32))
        else:
            sl = np.sqrt(((st[1:] - st[:-1]) ** 2).sum(1))
            seglen.append(np.concatenate([[0], sl]).astype(np.float32))
    total = float(sum(s.sum() for s in seglen)) + 1e-6
    times, dirs, acc = [], [], 0.0
    for st, sl in zip(strokes, seglen):
        t = ((acc + np.cumsum(sl)) / total).astype(np.float32)
        if len(st) == 1:
            d = np.array([0.5], np.float32)
        else:
            ang = np.arctan2(st[1:, 1] - st[:-1, 1], st[1:, 0] - st[:-1, 0])
            u = ((ang + np.pi) / (2 * np.pi)).astype(np.float32)
            d = np.concatenate([u, u[-1:]])
        times.append(t); dirs.append(d); acc += float(sl.sum())
    return times, dirs


def feature_color(t, d, ink=25):
    """Dark RGB (0-200) so a greyscale-pretrained stem still sees a stroke on white.
    t, d in [0, 1]; ink is the B-channel darkness (and the floor for R/G)."""
    ink = int(ink)
    r = int(np.clip(ink + t * 140, 0, 200))
    g = int(np.clip(ink + d * 140, 0, 200))
    b = int(np.clip(ink, 0, 200))
    return r, g, b


def render(strokes, rng: np.random.Generator | None = None, side=None, width=None, margin=4,
           hrange=(120, 320), wdiv=28, color=False, jitter=None, ink=None, hpx=None):
    """Render strokes to an RGB PIL image (white background, dark strokes).
    side: longer image side in px. The deepcopy renders have the longer side anywhere between
    ~100 and 512 px with 2-4 px strokes, i.e. the stroke width *relative to the formula*
    varies a lot after the model's fit-to-384 resize; by default we sample side log-uniformly.
    color=True writes time/direction/ink into R/G/B (see module docstring).
    jitter: subtle RGB perturbation (default on for greyscale train, off for colour).
    hpx: fixed formula height in px (used when side is None and we do not want a random size);
    eval uses hpx=200, width=2, ink=20, jitter=False."""
    rng = rng or np.random.default_rng(0)
    jitter = (not color) if jitter is None else jitter
    allp = np.concatenate(strokes)
    lo, hi = allp.min(0), allp.max(0)
    ext = np.maximum(hi - lo, 1.0)
    if side is None:
        # formula height ~120-320 px, longer side capped at 512 px (as in the deepcopy renders);
        # calibrated so the ink fraction after preprocessing matches deepcopy (0.057 vs 0.058)
        if hpx is None:
            hpx = float(np.exp(rng.uniform(np.log(hrange[0]), np.log(hrange[1]))))
        side = min(512.0, float(hpx) * ext.max() / ext[1])
    scale = (side - 2 * margin) / ext.max()
    W, H = (np.ceil(ext * scale) + 2 * margin).astype(int)
    if width is None:
        width = int(rng.choice([1, 2, 2, 2, 3]))  # calibrated: ink fraction ~ deepcopy renders
        width = max(1, min(width, int(round(ext[1] * scale / wdiv))))  # thin strokes for small glyphs
    S = 4  # sub-pixel precision for cv2 antialiasing
    if ink is None:
        ink = int(rng.integers(0, 60))
    if not color:
        img = np.full((H, W), 255, np.uint8)
        for st in strokes:
            q = ((st - lo) * scale + margin) * (1 << S)
            q = np.round(q).astype(np.int32).reshape(-1, 1, 2)
            if len(q) == 1:
                cv2.circle(img, tuple(q[0, 0]), max(1, width // 2) << S, int(ink), -1, cv2.LINE_AA, S)
            else:
                cv2.polylines(img, [q], False, int(ink), width, cv2.LINE_AA, S)
        rgb = np.repeat(img[:, :, None], 3, 2)
        if jitter and rng.random() < 0.5:  # subtle colour perturbation like the deepcopy renders
            rgb = np.clip(rgb.astype(np.int16) + rng.integers(-8, 9, size=3), 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)

    # colour: draw in BGR for cv2, convert to RGB for PIL
    img = np.full((H, W, 3), 255, np.uint8)
    times, dirs = stroke_features(strokes)
    rad = max(1, width // 2)
    for st, ts, ds in zip(strokes, times, dirs):
        q = np.round(((st - lo) * scale + margin) * (1 << S)).astype(np.int32)
        if len(q) == 1:
            r, g, b = feature_color(float(ts[0]), float(ds[0]), ink)
            cv2.circle(img, (int(q[0, 0]), int(q[0, 1])), rad << S, (b, g, r), -1, cv2.LINE_AA, S)
            continue
        for i in range(1, len(q)):
            r, g, b = feature_color(float(ts[i]), float(ds[i]), ink)
            cv2.line(img, (int(q[i - 1, 0]), int(q[i - 1, 1])), (int(q[i, 0]), int(q[i, 1])),
                     (b, g, r), width, cv2.LINE_AA, S)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def render_eval(strokes, color=True):
    """Deterministic eval render (fixed height 200 px, width 2, no jitter)."""
    return render(strokes, rng=np.random.default_rng(0), color=color, jitter=False,
                  ink=20, width=2, hpx=200)
