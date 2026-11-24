import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.decode import (  # noqa: E402
    ARRAY, BRACE, allowed_mask, append_missing_closers, closer_id, grammar_ids,
    stack_from_ids,
)


class FakeTok:
    """Minimal tokenizer stand-in: WordLevel ids matching Texo specials."""
    vocab_size = 16
    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    _map = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
            "{": 8, "}": 9, r"\begin{array}": 10, r"\end{array}": 11}

    def convert_tokens_to_ids(self, t):
        return self._map.get(t, self.unk_token_id)


def g():
    return grammar_ids(FakeTok())


def ids(*pieces, bos=True):
    m = FakeTok._map
    seq = [m["<s>"]] if bos else []
    seq += [m[p] for p in pieces]
    return seq


def allowed(seq, max_length=16):
    return allowed_mask(seq, FakeTok.vocab_size, g(), max_length)


def test_stack_tracks_braces_and_array():
    assert stack_from_ids(ids("{"), g()) == [BRACE]
    assert stack_from_ids(ids("{", "}"), g()) == []
    assert stack_from_ids(ids("{", "{", "}"), g()) == [BRACE]
    assert stack_from_ids(ids(r"\begin{array}"), g()) == [ARRAY]
    assert stack_from_ids(ids(r"\begin{array}", r"\end{array}"), g()) == []
    # closer of the wrong kind does not pop the other construct
    assert stack_from_ids(ids("{", r"\end{array}"), g())[-1] == "illegal"
    assert stack_from_ids(ids(r"\begin{array}", "}"), g())[-1] == "illegal"


def test_reject_closer_when_stack_empty():
    m = allowed(ids())
    assert m[FakeTok._map["}"]].item() is False
    assert m[FakeTok._map[r"\end{array}"]].item() is False
    assert m[FakeTok.eos_token_id].item() is True
    assert m[FakeTok._map["{"]].item() is True


def test_reject_mismatched_closer():
    m = allowed(ids("{"))
    assert m[FakeTok._map["}"]].item() is True
    assert m[FakeTok._map[r"\end{array}"]].item() is False
    assert m[FakeTok.eos_token_id].item() is False
    m = allowed(ids(r"\begin{array}"))
    assert m[FakeTok._map[r"\end{array}"]].item() is True
    assert m[FakeTok._map["}"]].item() is False
    assert m[FakeTok.eos_token_id].item() is False


def test_force_closer_near_max_length():
    # prefix length 3 (<s> { {), need 2 closers + EOS => remaining must be >= 3
    seq = ids("{", "{")
    # max_length == len(seq)+2  -> remaining 2, only closer
    m = allowed(seq, max_length=len(seq) + 2)
    assert m[FakeTok._map["}"]].item() is True
    assert m.sum().item() == 1
    # plenty of room: any token except mismatched closer / EOS still open
    m = allowed(seq, max_length=32)
    assert m[FakeTok._map["}"]].item() is True
    assert m[FakeTok.eos_token_id].item() is False
    assert m[FakeTok._map["{"]].item() is True


def test_eos_allowed_only_when_balanced():
    assert allowed(ids("{", "}"))[FakeTok.eos_token_id].item() is True
    assert allowed(ids("{"))[FakeTok.eos_token_id].item() is False


def test_nested_brace_then_array():
    seq = ids("{", r"\begin{array}")
    st = stack_from_ids(seq, g())
    assert st == [BRACE, ARRAY]
    m = allowed(seq)
    assert m[FakeTok._map[r"\end{array}"]].item() is True
    assert m[FakeTok._map["}"]].item() is False


def test_append_missing_closers():
    tok = FakeTok()
    # open brace and array, no EOS
    seq = ids("{", r"\begin{array}")
    # pad to length 5
    t = torch.tensor([seq + [tok.pad_token_id] * 2])
    out = append_missing_closers(t, tok)
    got = [i for i in out[0].tolist() if i != tok.pad_token_id]
    # must close array, then brace, then EOS
    assert got[-3:] == [tok._map[r"\end{array}"], tok._map["}"], tok.eos_token_id]


def test_closer_id():
    gg = g()
    assert closer_id(BRACE, gg) == FakeTok._map["}"]
    assert closer_id(ARRAY, gg) == FakeTok._map[r"\end{array}"]
