"""Decoding helpers for Texo/handwriter: length-normalised beam search and
simple LaTeX-grammar constraints (braces + \\begin{array}/\\end{array}).

Reusable across checkpoints (v2, v3, ...). HuggingFace beam search already
applies length_penalty (default 1.0) as score / length**alpha; pass 0.0 for
unnormalised log-probability ranking ("plain sum").

Brace/grammar constraints:
  * reject closers that do not match the current stack top
  * do not emit EOS while the stack is non-empty; force the matching closer
  * if remaining steps would not fit the closers + EOS, force the closer
"""
from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutput

OPEN_BRACE = "{"
CLOSE_BRACE = "}"
BEGIN_ARRAY = r"\begin{array}"
END_ARRAY = r"\end{array}"

# stack symbols
BRACE, ARRAY = "brace", "array"


def token_id(tok, piece: str):
    i = tok.convert_tokens_to_ids(piece)
    if i is None or i == tok.unk_token_id:
        return None
    return i


def grammar_ids(tok) -> dict:
    """Resolve the few token ids the constraint needs. Missing pieces are None."""
    return {
        "open": token_id(tok, OPEN_BRACE),
        "close": token_id(tok, CLOSE_BRACE),
        "begin": token_id(tok, BEGIN_ARRAY),
        "end": token_id(tok, END_ARRAY),
        "eos": tok.eos_token_id,
        "pad": tok.pad_token_id,
        "bos": tok.bos_token_id,
    }


def stack_from_ids(ids, g) -> list[str]:
    """Build the open-construct stack from a prefix (BOS/PAD/EOS ignored).

    An illegal closer (no matching opener) is recorded as 'illegal' so the
    caller can keep rejecting further closers; constrained decoding should
    never produce this state if applied from the first step.
    """
    stack: list[str] = []
    opener = {}
    if g["open"] is not None:
        opener[g["open"]] = BRACE
    if g["begin"] is not None:
        opener[g["begin"]] = ARRAY
    closer = {}
    if g["close"] is not None:
        closer[g["close"]] = BRACE
    if g["end"] is not None:
        closer[g["end"]] = ARRAY
    skip = {g["eos"], g["pad"], g["bos"]}
    for t in ids:
        if t in skip:
            if t == g["eos"]:
                break
            continue
        if t in opener:
            stack.append(opener[t])
        elif t in closer:
            want = closer[t]
            if stack and stack[-1] == want:
                stack.pop()
            else:
                stack.append("illegal")
    return stack


def closer_id(kind: str, g):
    return g["close"] if kind == BRACE else g["end"]


def allowed_mask(seq_ids, vocab_size: int, g: dict, max_length: int) -> torch.Tensor:
    """Boolean mask over the vocab for the next token of `seq_ids` (1d ints)."""
    mask = torch.ones(vocab_size, dtype=torch.bool)
    stack = stack_from_ids(seq_ids, g)
    # always reject a closer that does not match the top (or any closer if empty)
    for kind, cid in ((BRACE, g["close"]), (ARRAY, g["end"])):
        if cid is None:
            continue
        if not stack or stack[-1] != kind:
            mask[cid] = False

    remaining = max_length - len(seq_ids)
    n_close = sum(1 for s in stack if s in (BRACE, ARRAY))
    # illegal stack: don't force anything, just keep illegal closers off
    if any(s == "illegal" for s in stack):
        mask[g["eos"]] = True
        return mask

    if stack:
        mask[g["eos"]] = False
        # need n_close closers + 1 EOS; if we cannot afford a non-closer, force
        if remaining <= n_close + 1:
            mask.fill_(False)
            cid = closer_id(stack[-1], g)
            if cid is not None:
                mask[cid] = True
            if remaining > n_close:  # room for EOS only once the stack is empty, not yet
                pass
    return mask


class LatexGrammarLogitsProcessor(LogitsProcessor):
    """Reject unmatched closers; force closing of open braces/envs before EOS."""

    def __init__(self, tok, max_length: int):
        self.g = grammar_ids(tok)
        self.max_length = max_length
        self.vocab_size = tok.vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: (batch, seq); scores: (batch, vocab)
        ban = torch.finfo(scores.dtype).min
        vocab = scores.size(-1)
        finished = (input_ids == self.g["eos"]).any(dim=-1)
        for i, (row, done) in enumerate(zip(input_ids, finished)):
            if done:
                continue
            # drop trailing pad (should not be present on active hyps)
            seq = row.tolist()
            while seq and seq[-1] == self.g["pad"]:
                seq.pop()
            m = allowed_mask(seq, vocab, self.g, self.max_length)
            scores[i, ~m] = ban
        return scores


def append_missing_closers(ids_batch, tok, max_extra: int = 32):
    """Safety net: append matching closers + EOS if a hypothesis still has an
    open stack (e.g. hit max_length). ids_batch: (B, T) LongTensor or list of lists."""
    g = grammar_ids(tok)
    rows = ids_batch.tolist() if torch.is_tensor(ids_batch) else ids_batch
    out = []
    for seq in rows:
        seq = list(seq)
        while seq and seq[-1] == g["pad"]:
            seq.pop()
        if seq and seq[-1] == g["eos"]:
            body, has_eos = seq[:-1], True
        else:
            body, has_eos = seq, False
        stack = stack_from_ids(body, g)
        extra = []
        for kind in reversed(stack):
            if kind in (BRACE, ARRAY) and closer_id(kind, g) is not None and len(extra) < max_extra:
                extra.append(closer_id(kind, g))
        if extra or not has_eos:
            body = body + extra + [g["eos"]]
        else:
            body = seq
        out.append(body)
    # pad to a rectangular tensor
    m = max(len(r) for r in out)
    t = torch.full((len(out), m), g["pad"], dtype=torch.long)
    for i, r in enumerate(out):
        t[i, :len(r)] = torch.tensor(r, dtype=torch.long)
    return t


@torch.no_grad()
def encode(model, pixel_values):
    """Raw encoder last-hidden states for model.generate(encoder_outputs=...).

    Do not apply enc_to_dec_proj here: VisionEncoderDecoderModel.generate does
    that projection itself. (ONNX export is different — it bakes the proj in.)
    """
    return model.encoder(pixel_values=pixel_values).last_hidden_state


@torch.no_grad()
def generate(model, tok, pixel_values=None, encoder_hidden_states=None, device=None,
             max_length=256, num_beams=4, length_penalty=1.0, constrain=False,
             do_sample=False):
    """Beam / greedy generate. Pass either images or precomputed encoder states.

    length_penalty is HuggingFace's alpha (score / length**alpha). 1.0 is the
    library default used by scripts/eval.py and scripts/train.py; 0.0 ranks by
    raw sum of log-probs.
    """
    if encoder_hidden_states is None and pixel_values is None:
        raise ValueError("need pixel_values or encoder_hidden_states")
    if device is None:
        device = next(model.parameters()).device
    kw = dict(
        max_length=max_length,
        num_beams=num_beams,
        do_sample=do_sample,
        length_penalty=float(length_penalty),
        early_stopping=False,
        # when constraining, do not let ForcedEOS override a forced closer
        forced_eos_token_id=None if constrain else tok.eos_token_id,
    )
    if constrain:
        kw["logits_processor"] = LogitsProcessorList(
            [LatexGrammarLogitsProcessor(tok, max_length)]
        )
    if encoder_hidden_states is not None:
        h = encoder_hidden_states.to(device)
        kw["encoder_outputs"] = BaseModelOutput(last_hidden_state=h)
        # dummy mask so generate can infer the batch size consistently
        kw["attention_mask"] = torch.ones(h.size(0), 1, device=device, dtype=torch.long)
    else:
        kw["pixel_values"] = pixel_values.to(device)
    ids = model.generate(**kw)
    if constrain:
        ids = append_missing_closers(ids, tok)
        if ids.device != device:
            ids = ids.to(device)
    return tok.batch_decode(ids, skip_special_tokens=True)
