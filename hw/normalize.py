"""Canonical LaTeX label style for handwriter (used for training targets AND evaluation).

Canonical style = standard LaTeX as Texo/KaTeX users expect it:
  * function names as commands: MathWriting `lim_{x\\rightarrow0}sin(x)` -> `\\lim_{x\\rightarrow0}\\sin(x)`
  * standard arrow/symbol names: `\\infin`->`\\infty`, `\\rarr`->`\\rightarrow`, `\\to`->`\\rightarrow`,
    `\\ge`->`\\geq`, `\\le`->`\\leq`, `\\ne`->`\\neq`, `\\dag`->`\\dagger`, `\\empty`->`\\emptyset` ...
  * matrices as `\\begin{matrix}...\\end{matrix}` with explicit outer delimiters
    (`pmatrix` -> `(\\begin{matrix}..\\end{matrix})`, `array{cc}` -> `matrix`)
  * no whitespace, no spacing/sizing commands (`\\,` `\\quad` `\\left` `\\right` `\\big` `\\displaystyle`)
  * sub/superscript arguments always braced: `x_i` -> `x_{i}`; primes as `^{\\prime}`
  * `\\mathrm{..}`/`\\operatorname{..}`/`\\text{..}` unwrapped (function names -> commands)
  * `\\ldots`/`\\dots` -> `...`; `\\dfrac`/`\\tfrac` -> `\\frac`

normalize() is idempotent.  Use `tokens()` to get the canonical token list.
"""
import re

_TOK = re.compile(r"\\begin\{[a-zA-Z*]+\}|\\end\{[a-zA-Z*]+\}|\\[a-zA-Z]+|\\.|\S")

FUNCS = ["arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "coth", "sin", "cos", "tan",
         "cot", "sec", "csc", "log", "ln", "lg", "exp", "lim", "max", "min", "sup", "inf",
         "det", "gcd"]
_FUNC_RE = re.compile("|".join(sorted(FUNCS, key=len, reverse=True)))

SYN = {
    r"\infin": r"\infty", r"\rarr": r"\rightarrow", r"\to": r"\rightarrow",
    r"\Rarr": r"\Rightarrow", r"\implies": r"\Rightarrow", r"\larr": r"\leftarrow",
    r"\gets": r"\leftarrow", r"\harr": r"\leftrightarrow", r"\Harr": r"\Leftrightarrow",
    r"\iff": r"\Leftrightarrow", r"\empty": r"\emptyset", r"\varnothing": r"\emptyset",
    r"\dag": r"\dagger", r"\ge": r"\geq", r"\le": r"\leq", r"\ne": r"\neq",
    r"\geqslant": r"\geq", r"\leqslant": r"\leq", r"\lbrace": r"\{", r"\rbrace": r"\}",
    r"\vert": "|", r"\lvert": "|", r"\rvert": "|", r"\dfrac": r"\frac", r"\tfrac": r"\frac",
    r"\land": r"\wedge", r"\lor": r"\vee", r"\lnot": r"\neg", r"\ast": "*",
}
DROP = {r"\left", r"\right", r"\middle", r"\big", r"\Big", r"\bigg", r"\Bigg", r"\bigl", r"\bigr",
        r"\Bigl", r"\Bigr", r"\biggl", r"\biggr", r"\Biggl", r"\Biggr", r"\displaystyle",
        r"\textstyle", r"\scriptstyle", r"\,", r"\;", r"\:", r"\!", r"\ ", r"\quad", r"\qquad",
        r"\limits", r"\nolimits"}
UNWRAP = {r"\mathrm", r"\operatorname", r"\text", r"\textrm", r"\mathit", r"\textit", r"\rm"}
MATRIX_ENVS = {"matrix": None, "pmatrix": ("(", ")"), "bmatrix": ("[", "]"), "Bmatrix": (r"\{", r"\}"),
               "vmatrix": ("|", "|"), "Vmatrix": (r"\|", r"\|"), "array": None, "smallmatrix": None}


def _lex(s: str) -> list[str]:
    return _TOK.findall(s)


def _group_end(t: list[str], i: int) -> int:
    """t[i] == '{' -> index of matching '}' (or len(t)-1)."""
    depth = 0
    for j in range(i, len(t)):
        depth += t[j] == "{"
        depth -= t[j] == "}"
        if depth == 0:
            return j
    return len(t) - 1


def _funcs_in_letters(t: list[str]) -> list[str]:
    """Turn runs of single letters containing function names into commands."""
    out, i = [], 0
    while i < len(t):
        if len(t[i]) == 1 and t[i].isalpha():
            j = i
            while j < len(t) and len(t[j]) == 1 and t[j].isalpha():
                j += 1
            run = "".join(t[i:j])
            pos = 0
            for m in _FUNC_RE.finditer(run):
                out.extend(run[pos:m.start()])
                out.append("\\" + m.group())
                pos = m.end()
            out.extend(run[pos:])
            i = j
        else:
            out.append(t[i]); i += 1
    return out


def _pass(t: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(t):
        x = t[i]
        x = SYN.get(x, x)
        if x in DROP:
            i += 1; continue
        if x in (r"\ldots", r"\dots", r"\dotso", r"\dotsc"):
            out += [".", ".", "."]; i += 1; continue
        if x in UNWRAP and i + 1 < len(t) and t[i + 1] == "{":
            j = _group_end(t, i + 1)
            inner = _pass(t[i + 2:j])
            inner = [c for c in inner if c != "*"] if x == r"\operatorname" else inner
            out += _funcs_in_letters(inner)
            i = j + 1; continue
        if x.startswith(r"\begin{") or x.startswith(r"\end{"):
            env = x[x.index("{") + 1:-1].rstrip("*")
            if env in MATRIX_ENVS:
                delim = MATRIX_ENVS[env]
                if x.startswith(r"\begin"):
                    if delim: out.append(delim[0])
                    out.append(r"\begin{matrix}")
                    i += 1
                    if env == "array" and i < len(t) and t[i] == "{":  # drop column spec
                        i = _group_end(t, i) + 1
                    continue
                out.append(r"\end{matrix}")
                if delim: out.append(delim[1])
                i += 1; continue
        if x == "'":  # primes -> ^{\prime...}, merging with a following/preceding superscript
            k = i
            while k < len(t) and t[k] == "'":
                k += 1
            out += ["^", "{"] + [r"\prime"] * (k - i) + ["}"]
            i = k; continue
        if x in ("_", "^") and i + 1 < len(t):
            nxt = SYN.get(t[i + 1], t[i + 1])
            if nxt != "{":
                if nxt in DROP or nxt in UNWRAP or nxt.startswith(r"\begin"):
                    out.append(x); i += 1; continue  # rare; leave to later passes
                out += [x, "{", nxt, "}"]; i += 2; continue
        out.append(x); i += 1
    return out


def _merge_supers(t: list[str]) -> list[str]:
    """x^{\\prime}^{2} -> x^{\\prime2} is NOT done (ambiguous); only merge consecutive prime groups."""
    out = []
    for x in t:
        out.append(x)
        # pattern: ^ { \prime } ^ { \prime } -> ^ { \prime \prime }
        if len(out) >= 8 and out[-8:-4] == ["^", "{", r"\prime", "}"] and out[-4:-2] == ["^", "{"] \
                and out[-2] == r"\prime" and out[-1] == "}":
            del out[-5:-1]  # remove '} ^ { ' keep prime -> ^ { \prime \prime }
            out[-1:] = ["}"]
    return out


def tokens(latex: str) -> list[str]:
    t = _lex(latex)
    t = _pass(t)
    t = _funcs_in_letters(t)
    t = _merge_supers(t)
    return t


def detok(t: list[str]) -> str:
    out = ""
    for x in t:
        if out and re.search(r"\\[a-zA-Z]+$", out) and x[0].isalpha():
            out += " "
        out += x
    return out


def normalize(latex: str) -> str:
    return detok(tokens(latex))
