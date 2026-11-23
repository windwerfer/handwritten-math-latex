import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, VisionEncoderDecoderModel

from . import texo_vendor  # noqa: F401  registers the HGNetv2 encoder


def load(path):
    model = VisionEncoderDecoderModel.from_pretrained(path)
    tok = AutoTokenizer.from_pretrained(path)
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
def predict(model, tok, pixel_values, device, max_length=256, num_beams=1):
    ids = model.generate(pixel_values.to(device), max_length=max_length, num_beams=num_beams,
                         do_sample=False)
    return tok.batch_decode(ids, skip_special_tokens=True)
