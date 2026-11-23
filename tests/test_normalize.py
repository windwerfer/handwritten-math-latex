import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from hw.normalize import normalize, tokens
from hw.latex_tok import to_texo, from_texo

CASES = [
    (r"lim_{x\rightarrow0}sin(\frac{1}{x})", r"\lim_{x\rightarrow0}\sin(\frac{1}{x})"),
    (r"\lim_{x \to 0} \sin ( \frac { 1 } { x } )", r"\lim_{x\rightarrow0}\sin(\frac{1}{x})"),
    (r"lnx+xcos(y)", r"\ln x+x\cos(y)"),
    (r"\int_0^\infin f", r"\int_{0}^{\infty}f"),
    (r"a\ge b\le c\ne d", r"a\geq b\leq c\neq d"),
    (r"f'(x)+g''", r"f^{\prime}(x)+g^{\prime\prime}"),
    (r"x^{\prime}", r"x^{\prime}"),
    (r"(\begin{matrix}1&2\\3&4\end{matrix})", r"(\begin{matrix}1&2\\3&4\end{matrix})"),
    (r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", r"(\begin{matrix}1&2\\3&4\end{matrix})"),
    (r"\left( \begin{array} { c c } 1 & 2 \end{array} \right)", r"(\begin{matrix}1&2\end{matrix})"),
    (r"\mathrm{d}x", r"dx"),
    (r"\operatorname{sin} x", r"\sin x"),
    (r"a,\ldots,b", r"a,...,b"),
    (r"\dfrac{a}{b}\,\quad c", r"\frac{a}{b}c"),
    (r"x_\alpha^2", r"x_{\alpha}^{2}"),
    (r"\rarr \Rarr \larr \harr \Harr \empty \dag", r"\rightarrow\Rightarrow\leftarrow\leftrightarrow\Leftrightarrow\emptyset\dagger"),
    (r"dx dt", r"dxdt"),
    (r"\{a\}", r"\{a\}"),
    (r"inf_{x}f+tanh(x)", r"\inf_{x}f+\tanh(x)"),
    (r"liminf a_{n}", r"\lim\inf a_{n}"),
]


@pytest.mark.parametrize("src,want", CASES)
def test_normalize(src, want):
    assert normalize(src) == want


@pytest.mark.parametrize("src,want", CASES)
def test_idempotent(src, want):
    assert normalize(normalize(src)) == normalize(src)


@pytest.mark.parametrize("src,want", CASES)
def test_texo_roundtrip(src, want):
    assert normalize(from_texo(to_texo(normalize(src)))) == want


def test_texo_vocab_coverage():
    import json
    v = json.load(open(Path(__file__).resolve().parents[1] / "models/texo/tokenizer.json"))["model"]["vocab"]
    for src, _ in CASES:
        assert all(t in v for t in to_texo(src).split()), to_texo(src)
