import io
import math

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


def preprocess(img: Image.Image, size: int = 384, train: bool = False) -> torch.Tensor:
    """Texo eval preprocessing: crop white margin, fit into size x size keeping aspect,
    pad with black (as Texo does), grayscale, normalise. Light jitter when train=True."""
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
    a = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return torch.from_numpy(a).unsqueeze(0).repeat(3, 1, 1)


class MathWritingParquet(torch.utils.data.Dataset):
    """Rows: image (bytes), latex (MathWriting normalized label), sample_id."""

    def __init__(self, path, train=False, size=384, limit=None):
        df = pd.read_parquet(path)
        if limit:
            df = df.iloc[:limit]
        self.df = df.reset_index(drop=True)
        self.train, self.size = train, size

    def __len__(self):
        return len(self.df)

    def image(self, i) -> Image.Image:
        return Image.open(io.BytesIO(self.df.image_bytes[i]))

    def __getitem__(self, i):
        return {
            "pixel_values": preprocess(self.image(i), self.size, self.train),
            "text": to_texo(self.df.latex[i]),
            "latex": self.df.latex[i],
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
