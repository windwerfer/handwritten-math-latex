"""Canonical LaTeX (see hw/normalize.py) <-> Texo token sequences.

Texo's tokenizer is a WordLevel vocab (687 KaTeX tokens) with whitespace pre-tokenization.
A few canonical names are absent from that vocab; they are stored under Texo's synonym
(bijective map), so the model emits Texo names and from_texo() maps them back.
"""
from .normalize import detok, normalize, tokens

CANON_TO_TEXO = {
    r"\infty": r"\infin", r"\rightarrow": r"\rarr", r"\Rightarrow": r"\Rarr",
    r"\leftarrow": r"\larr", r"\Leftrightarrow": r"\Harr", r"\leftrightarrow": r"\harr",
    r"\emptyset": r"\empty", r"\dagger": r"\dag",
    r"\begin{matrix}": r"\begin{array}", r"\end{matrix}": r"\end{array}",
}
TEXO_TO_CANON = {v: k for k, v in CANON_TO_TEXO.items()}
# function commands missing from Texo's vocab -> \operatorname{letters} (unwrapped again by normalize)
OPNAME = {r"\tanh", r"\inf"}


def _texo_tok(t: str) -> list[str]:
    if t in OPNAME:
        return [r"\operatorname", "{", *t[1:], "}"]
    return [CANON_TO_TEXO.get(t, t)]


def tokenize(latex: str) -> list[str]:
    return tokens(latex)


def to_texo(latex: str) -> str:
    """any LaTeX -> canonical -> space-separated Texo tokens (training target)."""
    return " ".join(x for t in tokens(latex) for x in _texo_tok(t))


def from_texo(spaced: str) -> str:
    """Texo token string (model output) -> canonical LaTeX."""
    return normalize(detok([TEXO_TO_CANON.get(t, t) for t in spaced.split()]))
