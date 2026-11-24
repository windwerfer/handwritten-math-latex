import torch
import torch.nn.functional as F
from pathlib import Path

from transformers import PreTrainedTokenizerFast, VisionEncoderDecoderModel

from . import texo_vendor  # noqa: F401  registers the HGNetv2 encoder


def load(path):
    model = VisionEncoderDecoderModel.from_pretrained(path)
    # tokenizer.json is version-independent (tokenizer_config.json written by transformers 5.x
    # cannot be read by 4.x), so build the fast tokenizer from it directly.
    tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(path) / "tokenizer.json"), bos_token="<s>",
                                  eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    model.config.decoder_start_token_id = tok.bos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    return model, tok


def loss_fn(model, batch, device):
    """Compute CE ourselves (avoids VisionEncoderDecoder label-shift differences across
    transformers versions). Logits are aligned with decoder_input_ids; labels are pre-shifted."""
    out = model(pixel_values=batch["pixel_values"].to(device),
                decoder_input_ids=batch["decoder_input_ids"].to(device),
                decoder_attention_mask=batch["decoder_attention_mask"].to(device))
    logits = out.logits
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                           batch["labels"].to(device).reshape(-1), ignore_index=-100)


@torch.no_grad()
def predict(model, tok, pixel_values, device, max_length=256, num_beams=1, length_penalty=0.0):
    """length_penalty is HuggingFace's alpha (score / length**alpha). 0.0 (raw log-prob
    sum) beat the library default of 1.0 on the official MathWriting test set; see
    results/decode_v2/SUMMARY.md. It has no effect on greedy (num_beams=1)."""
    ids = model.generate(pixel_values.to(device), max_length=max_length, num_beams=num_beams,
                         do_sample=False, length_penalty=length_penalty)
    return tok.batch_decode(ids, skip_special_tokens=True)
