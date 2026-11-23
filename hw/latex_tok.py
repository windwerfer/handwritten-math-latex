"""LaTeX <-> Texo token-sequence conversion.

Texo's tokenizer is a WordLevel vocab (687 tokens, KaTeX-derived) with whitespace
pre-tokenization, so labels must be space-separated tokens ("\\frac { a } { b }").
MathWriting normalized labels are compact ("\\frac{a}{b}") and use some synonyms that
are not in Texo's vocab; we map them to Texo's spelling for training (TO_TEXO) and map
predictions back to MathWriting spelling for evaluation (TO_MW).
"""
import re

_TOK = re.compile(r"\\begin\{[a-z]+\}|\\end\{[a-z]+\}|\\[a-zA-Z]+|\\.|\S")

TO_TEXO = {
    r"\infty": r"\infin", r"\rightarrow": r"\rarr", r"\Rightarrow": r"\Rarr",
    r"\leftarrow": r"\larr", r"\Leftrightarrow": r"\Harr", r"\leftrightarrow": r"\harr",
    r"\emptyset": r"\empty", r"\dagger": r"\dag",
    r"\begin{matrix}": r"\begin{array}", r"\end{matrix}": r"\end{array}",
}
TO_MW = {v: k for k, v in TO_TEXO.items()}
TO_MW.update({r"\to": r"\rightarrow", r"\gets": r"\leftarrow", r"\iff": r"\Leftrightarrow",
              r"\implies": r"\Rightarrow",
              r"\varnothing": r"\emptyset", r"\dots": r"...", r"\ldots": r"...",
              r"\lbrace": r"\{", r"\rbrace": r"\}"})


def tokenize(latex: str) -> list[str]:
    return _TOK.findall(latex)


def to_texo(latex: str) -> str:
    """MathWriting label -> space-separated Texo token string (training target)."""
    return " ".join(TO_TEXO.get(t, t) for t in tokenize(latex))


def _drop_array_colspec(toks: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(toks):
        out.append(toks[i])
        if toks[i] == r"\begin{array}" and i + 1 < len(toks) and toks[i + 1] == "{":
            j = i + 2
            while j < len(toks) and toks[j] in ("c", "l", "r", "|"):
                j += 1
            if j < len(toks) and toks[j] == "}":
                i = j
        i += 1
    return out


def from_texo(spaced: str) -> str:
    """Texo token string -> compact LaTeX in MathWriting spelling (for eval/display)."""
    toks = _drop_array_colspec(spaced.split())
    toks = [TO_MW.get(t, t) for t in toks]
    out = ""
    for t in toks:
        if out and re.search(r"\\[a-zA-Z]+$", out) and t[0].isalpha():
            out += " "
        out += t
    return out


# synonyms treated as identical when scoring (applied to both prediction and reference)
EVAL_CANON = {r"\geq": r"\ge", r"\leq": r"\le", r"\neq": r"\ne", r"\to": r"\rightarrow",
              r"\gets": r"\leftarrow", r"\lbrace": r"\{", r"\rbrace": r"\}"}


def canon(latex: str) -> str:
    return "".join(EVAL_CANON.get(t, t) for t in tokenize(latex))
