#!/usr/bin/env python3
"""Like-for-like competitor eval on the official MathWriting test set.

All models are scored with the same hw.normalize canonicalisation and the same
metrics (exact match, CER, TER). A fixed random subset (default 1000, seed 0)
is used unless --full is set.

Examples:
  python scripts/eval_competitors.py --score-v2
  python scripts/eval_competitors.py --ink2latex --beams 1 5
  python scripts/eval_competitors.py --texteller
  python scripts/eval_competitors.py --score-v2 --ink2latex --texteller --beams 1 5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hw.metrics import scores  # noqa: E402
from hw.normalize import normalize  # noqa: E402

TEST_PARQUET = ROOT / "kaggle/dataset/test_full.parquet"
V2_PRED = {
    1: ROOT / "results/kaggle_v2_eval/eval/predictions_v2best_beam1.csv",
    4: ROOT / "results/kaggle_v2_eval/eval/predictions_v2best_beam4.csv",
}
OUT = ROOT / "results/competitors"
SUBSET_PATH = OUT / "subset.json"
INKML_TEST = ROOT / "data/raw/mathwriting-2024/test"
INK2LATEX_CKPT = ROOT / "models/competitors/ink2latex/r3.pt"
INK2LATEX_SRC = Path("/tmp/ink2latex")
TEXTELLER_DIR = ROOT / "models/competitors/texteller3-onnx"
SEED = 0
SUBSET_N = 1000


def official_ids() -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_table(TEST_PARQUET, columns=["sample_id"]).column("sample_id").to_pylist()


def official_labels() -> dict[str, str]:
    import pyarrow.parquet as pq
    t = pq.read_table(TEST_PARQUET, columns=["sample_id", "latex"])
    return dict(zip(t.column("sample_id").to_pylist(), t.column("latex").to_pylist()))


def load_or_make_subset(n: int, seed: int, full: bool) -> list[str]:
    ids = official_ids()
    if full or n >= len(ids):
        chosen = list(ids)
        meta = {"mode": "full", "n": len(chosen), "seed": None, "ids": chosen}
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(ids), size=n, replace=False))
        chosen = [ids[int(i)] for i in idx]
        meta = {"mode": "subset", "n": len(chosen), "seed": seed, "parent_n": len(ids), "ids": chosen}
    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBSET_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    return chosen


def write_metrics(name: str, metrics: dict) -> None:
    path = OUT / "metrics.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = metrics
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({name: metrics}, indent=2), flush=True)


def write_preds(name: str, rows: list[dict]) -> Path:
    path = OUT / f"predictions_{name}.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def score_v2(ids: list[str], labels: dict[str, str]) -> None:
    wanted = set(ids)
    for beam, path in V2_PRED.items():
        by_id = {}
        with path.open() as f:
            for row in csv.DictReader(f):
                if row["sample_id"] in wanted:
                    by_id[row["sample_id"]] = row["pred"]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"v2 beam{beam} missing {len(missing)} ids, e.g. {missing[:3]}")
        preds = [by_id[i] for i in ids]
        refs = [labels[i] for i in ids]
        m = scores(preds, refs)
        m.update({"model": "handwriter-v2best", "input": "image", "beam": beam,
                  "source": str(path.relative_to(ROOT)), "note": "rescored existing full-test predictions on this subset"})
        tag = f"v2best_beam{beam}"
        write_metrics(tag, m)
        write_preds(tag, [{"sample_id": i, "gt": labels[i], "pred": p} for i, p in zip(ids, preds)])


def run_ink2latex(ids: list[str], labels: dict[str, str], inkml_dir: Path, beams: list[int], device: str) -> None:
    if not INK2LATEX_CKPT.is_file():
        raise SystemExit(f"missing checkpoint {INK2LATEX_CKPT}")
    if not inkml_dir.is_dir():
        raise SystemExit(f"missing inkml dir {inkml_dir}")
    sys.path.insert(0, str(INK2LATEX_SRC))
    sys.path.insert(0, str(ROOT))
    import torch
    from hw.ink import parse_inkml, label_of
    from ink2latex.features import features
    from ink2latex.inference import greedy, beam_search, score_predictions, collate
    from ink2latex.tokenizer import to_latex, tokenize
    from ink2latex.training import load_checkpoint

    model, tokenizer, ckpt = load_checkpoint(str(INK2LATEX_CKPT), device)
    model.eval()
    cfg = ckpt.get("config") or {}
    print(f"ink2latex r3 loaded: {sum(p.numel() for p in model.parameters())} params, vocab {len(tokenizer)}, config {cfg}", flush=True)

    rows = []
    skipped = []
    for sid in ids:
        p = inkml_dir / f"{sid}.inkml"
        if not p.is_file():
            skipped.append(sid)
            continue
        ann, strokes = parse_inkml(p.read_text())
        pts = features(strokes)
        if len(pts) == 0:
            skipped.append(sid)
            continue
        raw_label = labels.get(sid) or label_of(ann)
        # Keep their official tokenisation (canonical braces) for their metric;
        # our metric uses the raw label through hw.normalize.
        their_tokens = tokenize(raw_label)
        token_ids = tokenizer.encode(their_tokens, allow_unknown=True)
        rows.append((pts, token_ids, {"id": sid, "tokens": their_tokens, "label": raw_label}))
    if skipped:
        print(f"ink2latex skipped {len(skipped)} / {len(ids)} (missing or empty)", flush=True)
    if not rows:
        raise SystemExit("no inkml rows to evaluate")

    for beam in beams:
        t0 = time.perf_counter()
        refs_their, preds_their, recs = [], [], []
        batch_size = 16 if beam == 1 else 8
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            points, mask, _, _ = collate(chunk, device)
            with torch.no_grad():
                if beam <= 1:
                    id_lists = greedy(model, points, mask)
                else:
                    id_lists = beam_search(model, points, mask, width=beam)
            for row, pred_ids in zip(chunk, id_lists):
                tok = tokenizer.decode(pred_ids)
                pred = to_latex(tok)
                refs_their.append(row[2]["tokens"])
                preds_their.append(tok)
                recs.append({"sample_id": row[2]["id"], "gt": row[2]["label"], "pred": pred,
                             "pred_tokens": tok})
            if (start // batch_size) % 10 == 0:
                print(f"  ink2latex beam{beam} {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
        elapsed = time.perf_counter() - t0
        ours = scores([r["pred"] for r in recs], [r["gt"] for r in recs])
        theirs = score_predictions(refs_their, preds_their)
        ours.update({
            "model": "ink2latex-r3",
            "input": "strokes",
            "beam": beam,
            "seconds": round(elapsed, 1),
            "sec_per_img": round(elapsed / max(len(recs), 1), 4),
            "device": device,
            "params": sum(p.numel() for p in model.parameters()),
            "skipped": len(skipped),
            "ink2latex_native_exact": theirs.get("exact_match"),
            "ink2latex_native_ter": theirs.get("token_error_rate"),
            "ink2latex_loose_exact": theirs.get("loose_exact_match"),
            "note": "native_* uses their tokenize()/canonical token exact and TER, not hw.normalize",
        })
        tag = f"ink2latex_r3_beam{beam}"
        write_metrics(tag, ours)
        write_preds(tag, recs)


def unwrap_display_math(s: str) -> str:
    """Strip TeXTeller-style display wrappers (\\[\\], $$, $) before scoring."""
    t = s.strip()
    for a, b in (("\\[", "\\]"), ("\\(", "\\)"), ("$$", "$$"), ("$", "$")):
        if t.startswith(a) and t.endswith(b) and len(t) >= len(a) + len(b):
            t = t[len(a):-len(b)].strip()
            break
    return t


def _texteller_preprocess(image_bytes: bytes, size=448, mean=0.9545467, std=0.15394445) -> np.ndarray:
    """Match TeXPen imagePreprocessing.ts: binarise, trim, letterbox to 448, normalize.

    Opaque images are thresholded at mean 128; the content box is trimmed and resized
    with the Hugging Face v2.Resize(size=447, max_size=448) rule, top-left padded on white.
    """
    from PIL import Image, ImageOps
    import io
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(im)
    # TeXPen opaque path: binary black/white by RGB sum > 384
    bin_mask = arr.sum(axis=2) <= 384
    work = np.full(arr.shape[:2], 255, np.uint8)
    work[bin_mask] = 0
    # trim white border (threshold 15 from top-left background, which is white)
    ink = work < 255 - 15
    if ink.any():
        ys, xs = np.where(ink)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        work = work[y0:y1, x0:x1]
    h, w = work.shape
    scale = min((size - 1) / max(min(w, h), 1), size / max(max(w, h), 1))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = Image.fromarray(work, mode="L").resize((new_w, new_h), Image.Resampling.BICUBIC)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(resized, (0, 0))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    x = (x - mean) / std
    return x[None, None]  # [1, 1, H, W]


def _inspect_onnx(path: Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return sess, [(i.name, i.shape, i.type) for i in sess.get_inputs()], [(o.name, o.shape, o.type) for o in sess.get_outputs()]


def _gpt2_byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2 / RoBERTa bytes_to_unicode, for ByteLevel BPE detokenisation."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def load_texteller_decoder(model_dir: Path):
    """Decode ids without the newer tokenizer.json format (tokenizers 0.19 cannot load it)."""
    obj = json.loads((model_dir / "tokenizer.json").read_text())
    id2tok = {int(v): k for k, v in obj["model"]["vocab"].items()}
    special = set()
    for a in obj.get("added_tokens") or []:
        id2tok[int(a["id"])] = a["content"]
        if a.get("special"):
            special.add(int(a["id"]))
    byte_dec = _gpt2_byte_decoder()

    def decode(ids: list[int]) -> str:
        pieces = [id2tok[i] for i in ids if i in id2tok and i not in special]
        text = "".join(pieces)
        try:
            return bytes(byte_dec[ch] for ch in text).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            return text

    return decode


def run_texteller(ids: list[str], labels: dict[str, str], model_dir: Path, max_new=128) -> None:
    enc_path = model_dir / "onnx" / "encoder_model_int8.onnx"
    dec_path = model_dir / "onnx" / "decoder_model_merged_int8.onnx"
    if not enc_path.is_file() or not dec_path.is_file():
        raise SystemExit(f"missing TexTeller ONNX under {model_dir}")
    import onnxruntime as ort
    import pyarrow.parquet as pq

    decode = load_texteller_decoder(model_dir)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    enc = ort.InferenceSession(str(enc_path), so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(dec_path), so, providers=["CPUExecutionProvider"])
    print("texteller encoder inputs", [(i.name, i.shape) for i in enc.get_inputs()],
          "outputs", [(o.name, o.shape) for o in enc.get_outputs()], flush=True)

    wanted = set(ids)
    images = {}
    pf = pq.ParquetFile(TEST_PARQUET)
    for batch in pf.iter_batches(columns=["sample_id", "image_bytes"], batch_size=256):
        d = batch.to_pydict()
        for sid, ib in zip(d["sample_id"], d["image_bytes"]):
            if sid in wanted:
                images[sid] = ib
    missing = [i for i in ids if i not in images]
    if missing:
        raise SystemExit(f"texteller missing images for {len(missing)} ids")

    enc_in = enc.get_inputs()[0].name
    enc_out = enc.get_outputs()[0].name
    dec_inputs = {i.name: i for i in dec.get_inputs()}
    dec_outputs = [o.name for o in dec.get_outputs()]
    past_names = [n for n in dec_inputs if n.startswith("past_key_values")]
    bos, eos, pad = 0, 2, 1
    print(f"texteller bos={bos} eos={eos} n_past={len(past_names)}", flush=True)

    def empty_past():
        feeds = {}
        for name in past_names:
            shape = dec_inputs[name].shape
            heads = shape[1] if isinstance(shape[1], int) else 16
            dim = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else 64
            feeds[name] = np.zeros((1, heads, 0, dim), np.float32)
        return feeds

    recs = []
    t0 = time.perf_counter()
    for n, sid in enumerate(ids, 1):
        px = _texteller_preprocess(images[sid])
        hidden = enc.run([enc_out], {enc_in: px})[0]
        # Full-prefix greedy (use_cache_branch=False). The merged-model KV-cache
        # path diverged after the first token in ORT; nocache matches step-0 logits.
        token_ids = [bos]
        empty = empty_past()
        for _step in range(max_new):
            feeds = {
                "input_ids": np.array([token_ids], np.int64),
                "encoder_hidden_states": hidden,
                "use_cache_branch": np.array([False]),
                **empty,
            }
            outs = dec.run(None, feeds)
            out_map = dict(zip(dec_outputs, outs))
            nxt = int(out_map["logits"][0, -1].argmax())
            token_ids.append(nxt)
            if nxt == eos:
                break
        raw = decode(token_ids).strip()
        text = unwrap_display_math(raw)
        recs.append({"sample_id": sid, "gt": labels[sid], "pred": text, "pred_raw": raw, "n_tokens": len(token_ids)})
        if n == 1 or n % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  texteller {n}/{len(ids)} ({elapsed/n:.2f}s/img) pred={text[:80]!r}", flush=True)
            if n == 1:
                print("  first gt =", labels[sid], flush=True)

    elapsed = time.perf_counter() - t0
    ours = scores([r["pred"] for r in recs], [r["gt"] for r in recs])
    ours.update({
        "model": "TeXPen-default/TexTeller3-ONNX-int8",
        "input": "image",
        "beam": 1,
        "quantization": "int8",
        "seconds": round(elapsed, 1),
        "sec_per_img": round(elapsed / max(len(recs), 1), 4),
        "max_new_tokens": max_new,
        "note": "TeXPen default profile (int8 enc+dec, greedy, nocache decode). Preprocess matches TeXPen (binarise, trim, letterbox 448). Display wrappers \\[ \\] are stripped before scoring.",
    })
    write_metrics("texteller3_int8_beam1", ours)
    write_preds("texteller3_int8_beam1", recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=SUBSET_N)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--full", action="store_true", help="Evaluate the official test set (7644), not a subset")
    ap.add_argument("--score-v2", action="store_true")
    ap.add_argument("--ink2latex", action="store_true")
    ap.add_argument("--texteller", action="store_true")
    ap.add_argument("--beams", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--inkml-dir", type=Path, default=INKML_TEST)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ids-limit", type=int, help="debug: only the first N of the chosen subset")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = load_or_make_subset(args.subset, args.seed, args.full)
    if args.ids_limit:
        ids = ids[: args.ids_limit]
        print(f"ids-limit applied, n={len(ids)}", flush=True)
    labels = official_labels()
    print(f"eval n={len(ids)} full={args.full} seed={None if args.full else args.seed}", flush=True)
    if args.score_v2:
        score_v2(ids, labels)
    if args.ink2latex:
        run_ink2latex(ids, labels, args.inkml_dir, args.beams, args.device)
    if args.texteller:
        run_texteller(ids, labels, TEXTELLER_DIR)


if __name__ == "__main__":
    main()
