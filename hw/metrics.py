from rapidfuzz.distance import Levenshtein

from .latex_tok import canon, tokenize


def _norm(s: str) -> str:
    # spaces carry no meaning in math mode; compare without them
    return canon(s)


def scores(preds: list[str], refs: list[str]) -> dict:
    """preds/refs: compact LaTeX in MathWriting spelling.
    cer: char-level edit distance / ref chars (corpus level).
    token_error_rate: same on LaTeX tokens. exact_match: whitespace-insensitive equality."""
    ce = cn = te = tn = em = 0
    for p, r in zip(preds, refs):
        pn, rn = _norm(p), _norm(r)
        ce += Levenshtein.distance(pn, rn); cn += len(rn)
        pt, rt = tokenize(p), tokenize(r)
        te += Levenshtein.distance(pt, rt); tn += len(rt)
        em += pn == rn
    return {"n": len(refs), "cer": ce / max(cn, 1), "token_error_rate": te / max(tn, 1),
            "exact_match": em / max(len(refs), 1)}
