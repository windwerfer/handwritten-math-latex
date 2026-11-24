import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw.average import average_state_dicts, discover_epoch_checkpoints, save_averaged  # noqa: E402


def test_average_floats_only(tmp_path):
    a = {"w": torch.tensor([0.0, 2.0]), "i": torch.tensor([3], dtype=torch.int64)}
    b = {"w": torch.tensor([2.0, 4.0]), "i": torch.tensor([3], dtype=torch.int64)}
    out = average_state_dicts([a, b])
    assert torch.allclose(out["w"], torch.tensor([1.0, 3.0]))
    assert out["i"].dtype == torch.int64 and int(out["i"]) == 3


def test_discover_and_save(tmp_path):
    run = tmp_path / "run"
    for name, val in (("best", 0.0), ("last", 2.0)):
        d = run / name
        d.mkdir(parents=True)
        torch.save({"w": torch.tensor([val])}, d / "pytorch_model.bin")
        (d / "config.json").write_text("{}")
    found = discover_epoch_checkpoints([tmp_path, run])
    # both the parent and the run dir itself may match; the run dir is the one with 2 snaps
    snaps = found[str(run)]
    assert {p.name for p in snaps} == {"best", "last"}
    out = tmp_path / "avg"
    save_averaged(snaps, out)
    if (out / "model.safetensors").exists():
        from safetensors.torch import load_file
        blob = load_file(str(out / "model.safetensors"))
    else:
        blob = torch.load(out / "pytorch_model.bin", map_location="cpu", weights_only=True)
    assert torch.allclose(blob["w"], torch.tensor([1.0]))
    assert (out / "config.json").exists()
    assert (out / "average.json").exists()


def test_discover_ignores_single_and_flat(tmp_path):
    # a lone best/ is not enough to average
    (tmp_path / "run" / "best").mkdir(parents=True)
    torch.save({"w": torch.tensor([1.0])}, tmp_path / "run" / "best" / "pytorch_model.bin")
    assert discover_epoch_checkpoints(tmp_path / "run") == {}
    # flat v1__ / v2best__ files are different models, not epochs
    (tmp_path / "flat").mkdir()
    torch.save({"w": torch.tensor([0.0])}, tmp_path / "flat" / "v2best__model.safetensors")
    torch.save({"w": torch.tensor([1.0])}, tmp_path / "flat" / "v1__model.safetensors")
    assert discover_epoch_checkpoints(tmp_path / "flat") == {}
