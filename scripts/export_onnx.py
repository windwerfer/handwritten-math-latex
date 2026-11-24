"""Export a fine-tuned Texo checkpoint to ONNX (encoder + decoder) in fp32 / fp16 / int8,
and check parity against PyTorch greedy decoding on test samples.

  python scripts/export_onnx.py --model runs/x/best --out onnx/v2 --test data/subset/test.parquet --n 200

Graphs (no KV cache; the 2-layer decoder is cheap enough to re-run on the prefix):
  encoder.onnx : pixel_values f32[B,3,384,384]          -> encoder_hidden_states f32[B,144,384]
  decoder.onnx : input_ids i64[B,T], encoder_hidden_states -> logits f32[B,T,687]
Decoding: start with <s>=0, append argmax of logits[:, -1] until </s>=2 (or max_len).
Pre-processing: hw/data.py:preprocess (crop margin, fit 384x384, pad black, gray, (x/255-0.7931)/0.1738).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.data import MathWritingParquet  # noqa: E402
from hw.latex_tok import from_texo  # noqa: E402
from hw.metrics import scores  # noqa: E402
from hw.model import load, predict  # noqa: E402


class Enc(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, pixel_values):
        h = self.m.encoder(pixel_values=pixel_values).last_hidden_state
        if getattr(self.m, "enc_to_dec_proj", None) is not None:
            h = self.m.enc_to_dec_proj(h)
        return h


class Dec(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, input_ids, encoder_hidden_states):
        return self.m.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
                              use_cache=False, return_dict=True).logits


def export(model, out: Path):
    model.eval()
    px = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        h = Enc(model)(px)
    torch.onnx.export(Enc(model), (px,), out / "encoder.onnx", input_names=["pixel_values"],
                      output_names=["encoder_hidden_states"], opset_version=17, dynamo=False,
                      dynamic_axes={"pixel_values": {0: "batch"}, "encoder_hidden_states": {0: "batch"}})
    ids = torch.zeros(1, 5, dtype=torch.long)
    torch.onnx.export(Dec(model), (ids, h), out / "decoder.onnx", input_names=["input_ids", "encoder_hidden_states"],
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                    "encoder_hidden_states": {0: "batch"}, "logits": {0: "batch", 1: "seq"}})


def make_variants(d: Path):
    import onnx
    from onnxruntime.transformers import float16
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for part in ("encoder", "decoder"):
        m = onnx.load(d / f"{part}.onnx")
        onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), d / f"{part}_fp16.onnx")
        quantize_dynamic(d / f"{part}.onnx", d / f"{part}_int8.onnx", weight_type=QuantType.QInt8)


class OnnxTexo:
    def __init__(self, enc, dec, threads=4):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        self.enc = ort.InferenceSession(str(enc), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(dec), so, providers=["CPUExecutionProvider"])

    def greedy(self, px: np.ndarray, max_len=256):
        h = self.enc.run(None, {"pixel_values": px.astype(np.float32)})[0]
        B = px.shape[0]
        ids = np.zeros((B, 1), dtype=np.int64)
        done = np.zeros(B, bool)
        for _ in range(max_len - 1):
            logits = self.dec.run(None, {"input_ids": ids, "encoder_hidden_states": h})[0]
            nxt = logits[:, -1].argmax(-1)
            nxt[done] = 1  # pad
            ids = np.concatenate([ids, nxt[:, None]], 1)
            done |= nxt == 2
            if done.all():
                break
        return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", nargs="+", default=["data/subset/test.parquet"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model, tok = load(a.model); model.eval()
    ds = MathWritingParquet(a.test, limit=a.n)
    px = torch.stack([ds[i]["pixel_values"] for i in range(len(ds))])
    gts = [str(x) for x in ds.latex]
    res = {}
    t0 = time.time(); pt = []
    for i in range(0, len(ds), a.batch):
        pt += predict(model, tok, px[i:i + a.batch], torch.device("cpu"))
    pt_pred = [from_texo(p) for p in pt]
    res["pytorch_fp32"] = {**scores(pt_pred, gts), "sec_per_img": (time.time() - t0) / len(ds)}
    print(res["pytorch_fp32"], flush=True)
    # export AFTER the PyTorch reference run (tracing mutates module state)
    export(model, out); make_variants(out)
    tok.save_pretrained(out)  # tokenizer.json for C++/Rust (tokenizers crate / plain vocab lookup)
    for name, enc, dec in [("onnx_fp32", "encoder", "decoder"), ("onnx_fp16", "encoder_fp16", "decoder_fp16"),
                           ("onnx_int8", "encoder_int8", "decoder_int8"),
                           ("onnx_enc_fp32_dec_int8", "encoder", "decoder_int8"),
                           ("onnx_enc_fp16_dec_int8", "encoder_fp16", "decoder_int8")]:
        try:
            sess = OnnxTexo(out / f"{enc}.onnx", out / f"{dec}.onnx")
            t0 = time.time(); preds = []
            for i in range(0, len(ds), a.batch):
                ids = sess.greedy(px[i:i + a.batch].numpy())
                preds += tok.batch_decode(ids, skip_special_tokens=True)
            pp = [from_texo(p) for p in preds]
            r = scores(pp, gts)
            r["sec_per_img"] = (time.time() - t0) / len(ds)
            r["identical_to_pytorch"] = float(np.mean([x == y for x, y in zip(pp, pt_pred)]))
            r["size_mb"] = round(sum(os.path.getsize(out / f"{p}.onnx") for p in (enc, dec)) / 1e6, 1)
        except Exception as e:  # noqa: BLE001
            r = {"error": repr(e)[:500]}
        res[name] = r
        print(name, r, flush=True)
    res["files_mb"] = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(out.glob("*.onnx"))}
    (out / "parity.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
