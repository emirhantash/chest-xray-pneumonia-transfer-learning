"""Tests for src/checkpoint.py."""

import random

import numpy as np
import torch

from src.checkpoint import get_rng_state, load_checkpoint, save_checkpoint, set_rng_state


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(3, 2)
    state = {"epoch": 4, "model": model.state_dict(), "history": [{"epoch": 1}], "note": "x"}
    path = tmp_path / "last.pt"
    save_checkpoint(state, path)

    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 4 and loaded["history"] == [{"epoch": 1}]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded["model"][key], value)
    # The temporary file is renamed, not left behind.
    assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]


def test_save_checkpoint_replaces_an_existing_file(tmp_path):
    path = tmp_path / "last.pt"
    save_checkpoint({"epoch": 1}, path)
    save_checkpoint({"epoch": 2}, path)
    assert load_checkpoint(path)["epoch"] == 2


def _draw():
    return random.random(), float(np.random.rand()), torch.rand(4)


def test_rng_state_roundtrip(tmp_path):
    torch.manual_seed(0)
    state = get_rng_state()
    save_checkpoint({"rng": state}, tmp_path / "rng.pt")
    expected = _draw()

    _draw()  # advance every generator
    set_rng_state(load_checkpoint(tmp_path / "rng.pt")["rng"])
    actual = _draw()
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])
