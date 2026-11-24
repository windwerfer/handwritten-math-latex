"""Average the weights of several checkpoints of the same run (SWA-style).

Only floating-point tensors are averaged; integer buffers are copied from the
first checkpoint. Does not invent checkpoints: the caller (or
discover_epoch_checkpoints) has to point at real epoch/best/last directories
that all belong to one training run.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")


def _weight_file(path: Path) -> Path | None:
    for n in WEIGHT_NAMES:
        p = path / n
        if p.exists():
            return p
    return None


def load_state(path) -> dict:
    path = Path(path)
    wf = _weight_file(path)
    if wf is None:
        raise FileNotFoundError(f"no model weights in {path}")
    if wf.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(wf))
    blob = torch.load(wf, map_location="cpu", weights_only=True)
    return blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob


def average_state_dicts(states: list[dict]) -> dict:
    if not states:
        raise ValueError("no state dicts to average")
    keys = states[0].keys()
    for i, st in enumerate(states[1:], 1):
        if st.keys() != keys:
            missing = keys - st.keys()
            extra = st.keys() - keys
            raise ValueError(f"checkpoint {i} keys differ (missing {missing}, extra {extra})")
    out = {}
    n = float(len(states))
    for k, v0 in states[0].items():
        if v0.is_floating_point():
            acc = v0.float().clone()
            for st in states[1:]:
                acc += st[k].float()
            out[k] = (acc / n).to(dtype=v0.dtype)
        else:
            out[k] = v0.clone()
    return out


def save_averaged(paths, out_dir) -> Path:
    """Average `paths` (checkpoint directories) and write a loadable HF dir to out_dir."""
    paths = [Path(p) for p in paths]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    states = [load_state(p) for p in paths]
    avg = average_state_dicts(states)
    # write weights
    try:
        from safetensors.torch import save_file
        save_file(avg, str(out / "model.safetensors"), metadata={"format": "pt"})  # HF from_pretrained needs it
    except Exception:  # noqa: BLE001
        torch.save(avg, out / "pytorch_model.bin")
    # copy tokenizer + config from the first checkpoint
    first = paths[0]
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json", "README.md"):
        src = first / name
        if src.exists():
            shutil.copy2(src, out / name)
    meta = {"averaged_from": [str(p) for p in paths], "n": len(paths)}
    (out / "average.json").write_text(json.dumps(meta, indent=2))
    return out


# Directory names we treat as epoch/best/last snapshots of one run.
_SNAPSHOT = {"best", "last"}


def discover_epoch_checkpoints(roots) -> dict:
    """Look under each root for sibling checkpoints of the same run.

    Returns a dict of run_name -> [paths] only for runs that have 2+ weight
    directories (so averaging is possible). Flat Kaggle dataset files like
    v1__model.safetensors / v2best__model.safetensors are different models,
    not epochs, and are ignored.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    found: dict[str, list[Path]] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        # case 1: root itself is a run dir containing best/ last/ epoch_*/
        snaps = _snapshots_in(root)
        if len(snaps) >= 2:
            found[str(root)] = snaps
            continue
        # case 2: root contains one or more run dirs
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            snaps = _snapshots_in(child)
            if len(snaps) >= 2:
                found[str(child)] = snaps
    return found


def _snapshots_in(run: Path) -> list[Path]:
    out = []
    if not run.is_dir():
        return out
    for p in sorted(run.iterdir()):
        if not p.is_dir():
            continue
        name = p.name.lower()
        interesting = (
            name in _SNAPSHOT
            or name.startswith("epoch")
            or name.startswith("ep")
            or name.startswith("checkpoint")
        )
        if interesting and _weight_file(p) is not None:
            out.append(p)
    # also accept the run dir itself if it is a single checkpoint (not useful alone)
    return out
