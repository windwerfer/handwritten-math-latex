import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from .latex_tok import to_texo

MEAN, STD = 0.7931, 0.1738  # UniMERNet/Texo normalisation (grayscale)


def crop_margin(img: Image.Image) -> Image.Image:
    a = np.asarray(img.convert("L"), dtype=np.float32)
    lo, hi = a.min(), a.max()
    if hi == lo:
        return img
    mask = ((a - lo) / (hi - lo) * 255) < 200
    ys, xs = np.nonzero(mask)
    return img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def preprocess(img: Image.Image, size: int = 384, train: bool = False, color: bool = False) -> torch.Tensor:
    """Texo eval preprocessing: crop white margin, fit into size x size keeping aspect,
    pad with black (as Texo does), normalise. Light geometric jitter when train=True.
    color=False (default, v1-v3): convert to greyscale and repeat to 3 channels.
    color=True: keep RGB so stroke-time/direction channels survive (same MEAN/STD)."""
    img = img.convert("RGB")
    img = crop_margin(img)
    if train:
        s = np.random.uniform(0.85, 1.0)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC)
        img = img.rotate(np.random.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    r = min(size / img.width, size / img.height)
    img = img.resize((max(1, round(img.width * r)), max(1, round(img.height * r))), Image.BICUBIC)
    dw, dh = size - img.width, size - img.height
    img = ImageOps.expand(img, (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2), fill=0)
    if color:
        a = np.asarray(img, dtype=np.float32) / 255.0  # H,W,3
        a = (a - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
    a = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return torch.from_numpy(a).unsqueeze(0).repeat(3, 1, 1)


class MathWritingParquet(torch.utils.data.Dataset):
    """MathWriting rows from one or more parquet files.
    Accepts either our subset format (image_bytes column) or the HF
    deepcopy/MathWriting-human format (image struct column). limit -> seeded random sample.

    Memory: all images live in ONE contiguous uint8 buffer + offsets, labels in fixed-width
    numpy string arrays. No per-sample Python objects are kept, so forked DataLoader workers
    don't gradually copy the dataset through refcount writes (copy-on-access), which is what
    OOM-killed the v2 Kaggle run after ~7 h with 230k samples held as pandas bytes objects."""

    def __init__(self, paths, train=False, size=384, limit=None, seed=0, color=False):
        import pyarrow.parquet as pq
        if isinstance(paths, (str, Path)):
            paths = [paths]
        blobs, latex, sids = [], [], []
        for p in paths:
            t = pq.read_table(p)
            col = t.column("image_bytes") if "image_bytes" in t.column_names else \
                t.column("image").combine_chunks().field("bytes")
            blobs += col.to_pylist()
            latex += t.column("latex").to_pylist()
            sids += t.column("sample_id").to_pylist()
            del t, col
        idx = np.arange(len(latex))
        if limit and limit < len(idx):
            idx = np.random.RandomState(seed).choice(idx, size=limit, replace=False)
        lens = np.array([len(blobs[i]) for i in idx], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(lens)])
        self.buf = np.frombuffer(b"".join(blobs[i] for i in idx), dtype=np.uint8)
        del blobs
        self.latex = np.array([latex[i] for i in idx])          # fixed-width unicode, no objects
        self.texo = np.array([to_texo(latex[i]) for i in idx])  # precomputed training targets
        self.sample_id = np.array([sids[i] for i in idx])
        self.train, self.size, self.color = train, size, color

    def __len__(self):
        return len(self.latex)

    def image(self, i) -> Image.Image:
        a, b = self.offsets[i], self.offsets[i + 1]
        return Image.open(io.BytesIO(self.buf[a:b].tobytes()))

    def __getitem__(self, i):
        return {
            "pixel_values": preprocess(self.image(i), self.size, self.train, color=self.color),
            "text": str(self.texo[i]),
            "latex": str(self.latex[i]),
            "idx": i,
        }


def make_collate(tokenizer, max_len=256):
    pad = tokenizer.pad_token_id

    def collate(batch):
        enc = tokenizer([b["text"] for b in batch], add_special_tokens=True, padding="longest",
                        truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"]
        # Texo convention: decoder_input_ids = <s> ... </s>; labels = ids shifted left.
        labels = torch.full_like(ids, -100)
        labels[:, :-1] = ids[:, 1:]
        labels[labels == pad] = -100
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "decoder_input_ids": ids,
            "decoder_attention_mask": enc["attention_mask"],
            "labels": labels,
            "latex": [b["latex"] for b in batch],
            "idx": [b["idx"] for b in batch],
        }

    return collate


class InkArrays:
    """Read-only access to inks packed by scripts/prep_inks.py (np.load mmap, opened lazily
    in each worker; file-backed pages are shared, so no copy-on-write growth)."""

    def __init__(self, ink_dir, split):
        self.dir, self.split = Path(ink_dir), split
        meta = pd.read_parquet(self.dir / f"{split}_meta.parquet")
        self.sample_id = meta["sample_id"].to_numpy().astype(str)
        self.latex = meta["latex"].to_numpy().astype(str)
        self.texo = np.array([to_texo(s) for s in self.latex])
        self._m = None

    def _open(self):
        if self._m is None:
            ld = lambda n: np.load(self.dir / f"{self.split}_{n}.npy", mmap_mode="r")  # noqa: E731
            self._m = (ld("points"), ld("soff"), ld("ioff"))
        return self._m

    def __len__(self):
        return len(self.latex)

    def strokes(self, i):
        pts, soff, ioff = self._open()
        a, b = ioff[i], ioff[i + 1]
        return [np.asarray(pts[soff[k]:soff[k + 1]]) for k in range(a, b)]

    def __getstate__(self):  # never pickle open memmaps into workers
        d = dict(self.__dict__); d["_m"] = None
        return d


class MixedTrain(torch.utils.data.Dataset):
    """Index space: [0, n_human) = human samples, [n_human, n_human + n_synth) = synthetic inks.
    Human sample: with prob human_image_prob the deepcopy-rendered image (as in v1/v2), otherwise
    its ink re-rendered with stroke augmentation (if the ink is available). Synthetic samples are
    always ink-rendered with stroke augmentation."""

    def __init__(self, human: "MathWritingParquet", human_inks: InkArrays | None, synth: InkArrays | None,
                 human_image_prob=0.5, aug_strength=1.0, size=384, color=False):
        self.human, self.synth, self.hinks = human, synth, human_inks
        self.n_h, self.n_s = len(human), (len(synth) if synth is not None else 0)
        # colour channels need InkML strokes; deepcopy images have none, so always re-render
        self.color = color
        self.p_img, self.strength, self.size = (0.0 if color else human_image_prob), aug_strength, size
        self.h2ink = np.full(self.n_h, -1, np.int64)
        if human_inks is not None:
            pos = {s: i for i, s in enumerate(human_inks.sample_id)}
            self.h2ink = np.array([pos.get(str(s), -1) for s in human.sample_id], np.int64)
            del pos

    def __len__(self):
        return self.n_h + self.n_s

    def _ink_item(self, inks, j, i):
        from .ink import augment_strokes, render
        rng = np.random.default_rng()
        img = render(augment_strokes(inks.strokes(j), rng, self.strength), rng, color=self.color)
        return {"pixel_values": preprocess(img, self.size, train=True, color=self.color),
                "text": str(inks.texo[j]), "latex": str(inks.latex[j]), "idx": i}

    def __getitem__(self, i):
        if i >= self.n_h:
            return self._ink_item(self.synth, i - self.n_h, i)
        j = self.h2ink[i]
        if j >= 0 and np.random.random() >= self.p_img:
            return self._ink_item(self.hinks, j, i)
        return self.human[i]


class MixSampler(torch.utils.data.Sampler):
    """Each epoch: every human sample once + synth_ratio * n_human synthetic samples drawn
    without replacement (fresh draw each epoch), shuffled together."""

    def __init__(self, ds: MixedTrain, synth_ratio=1.0, seed=0):
        self.ds, self.ratio, self.epoch, self.seed = ds, synth_ratio, 0, seed
        self.n_syn = min(ds.n_s, int(round(ds.n_h * synth_ratio)))

    def __len__(self):
        return self.ds.n_h + self.n_syn

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch); self.epoch += 1
        syn = self.ds.n_h + rng.choice(self.ds.n_s, size=self.n_syn, replace=False) if self.n_syn else np.array([], int)
        idx = np.concatenate([np.arange(self.ds.n_h), syn]); rng.shuffle(idx)
        return iter(idx.tolist())

class InkEval(torch.utils.data.Dataset):
    """Official valid/test rows (labels and sample_ids from the parquet split) rendered
    from packed InkML strokes. Used when training a colour-stroke model so eval images
    live in the same domain. Raises if coverage of the official split is incomplete."""

    def __init__(self, parquet_paths, inks: InkArrays, color=True, size=384, limit=None, seed=0):
        official = MathWritingParquet(parquet_paths, limit=limit, seed=seed)
        pos = {str(s): i for i, s in enumerate(inks.sample_id)}
        self.ink_i = np.array([pos.get(str(s), -1) for s in official.sample_id], np.int64)
        missing = int((self.ink_i < 0).sum())
        if missing:
            raise RuntimeError(f"InkEval: {missing}/{len(official)} official samples have no ink "
                               f"in {inks.split} (dir {inks.dir})")
        self.official, self.inks = official, inks
        self.color, self.size = color, size
        self.latex = official.latex
        self.sample_id = official.sample_id

    def __len__(self):
        return len(self.official)

    def image(self, i) -> Image.Image:
        from .ink import render_eval
        return render_eval(self.inks.strokes(int(self.ink_i[i])), color=self.color)

    def __getitem__(self, i):
        return {
            "pixel_values": preprocess(self.image(i), self.size, train=False, color=self.color),
            "text": str(self.official.texo[i]),
            "latex": str(self.official.latex[i]),
            "idx": i,
        }
