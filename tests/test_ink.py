import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.ink import augment_strokes, parse_inkml, render  # noqa: E402

INK = """<ink xmlns="http://www.w3.org/2003/InkML">
<annotation type="label">x^2</annotation><annotation type="normalizedLabel">x^{2}</annotation>
<annotation type="sampleId">00ff</annotation>
<traceFormat><channel name="X" type="decimal" /><channel name="Y" type="decimal" /></traceFormat>
<trace id="0">0 0 1,10 10 2,20 20 3</trace><trace id="1">20 0 4,0 20 5</trace>
<trace id="2">25 -10 6,30 -15 7,35 -10 8,25 0 9,35 0 10</trace></ink>"""


def test_parse():
    ann, st = parse_inkml(INK)
    assert ann["normalizedLabel"] == "x^{2}" and ann["sampleId"] == "00ff"
    assert [len(s) for s in st] == [3, 2, 5] and st[0].dtype == np.float32
    assert np.allclose(st[0][-1], [20, 20])


def test_augment_small_and_shape_preserving():
    _, st = parse_inkml(INK)
    rng = np.random.default_rng(0)
    for _ in range(50):
        aug = augment_strokes(st, rng)
        assert [len(s) for s in aug] == [len(s) for s in st]
        a, b = np.concatenate(st), np.concatenate(aug)
        assert np.isfinite(b).all()
        h = np.ptp(a[:, 1])
        assert np.abs(b - a).max() < 0.5 * h  # small perturbation relative to formula size
    assert not np.allclose(np.concatenate(augment_strokes(st, rng)), np.concatenate(st))


def test_render():
    _, st = parse_inkml(INK)
    img = render(st, np.random.default_rng(0), side=300, width=2)
    a = np.asarray(img.convert("L"))
    assert max(a.shape) == 300 and a.mean() > 200 and a.min() < 80  # white bg, dark ink
    tiny = render([np.array([[5, 5]], np.float32)], np.random.default_rng(0), side=64, width=2)
    assert np.asarray(tiny.convert("L")).min() < 128  # single-point stroke is drawn
