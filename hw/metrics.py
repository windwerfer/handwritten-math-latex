from rapidfuzz.distance import Levenshtein

from .normalize import tokens


def scores(preds: list[str], refs: list[str]) -> dict:
    """Both sides are canonicalised with hw.normalize before scoring.
    cer: char edit distance / ref chars over the canonical strings (corpus level);
    token_error_rate: same over canonical tokens; exact_match: canonical strings equal."""
    ce = cn = te = tn = em = 0
    for p, r in zip(preds, refs):
        pt, rt = tokens(p), tokens(r)
        ps, rs = "".join(pt), "".join(rt)
        ce += Levenshtein.distance(ps, rs); cn += len(rs)
        te += Levenshtein.distance(pt, rt); tn += len(rt)
        em += ps == rs
    return {"n": len(refs), "cer": ce / max(cn, 1), "token_error_rate": te / max(tn, 1),
            "exact_match": em / max(len(refs), 1)}
