"""Tests for src/utils.py and the baseline configuration."""

import random

import numpy as np
import pytest
import torch

from src.utils import get_device, load_config, set_seed
from tests.conftest import BASELINE_CONFIG


def _draw():
    return random.random(), np.random.rand(), torch.rand(3)


def test_set_seed_is_reproducible():
    generator_a = set_seed(123)
    draws_a = _draw()
    shuffled_a = torch.randperm(10, generator=generator_a)

    generator_b = set_seed(123)
    draws_b = _draw()
    shuffled_b = torch.randperm(10, generator=generator_b)

    assert draws_a[:2] == draws_b[:2]
    torch.testing.assert_close(draws_a[2], draws_b[2])
    torch.testing.assert_close(shuffled_a, shuffled_b)


def test_get_device_returns_a_supported_device():
    assert get_device().type in {"cuda", "mps", "cpu"}


def test_load_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 1\ntrain:\n  lr: 0.01\n", encoding="utf-8")
    assert load_config(path) == {"seed": 1, "train": {"lr": 0.01}}


def test_baseline_config_matches_protocol():
    # Guards the baseline protocol (interim report, Table 3) against accidental edits.
    cfg = load_config(BASELINE_CONFIG)
    assert cfg["seed"] == 42
    assert cfg["data"]["image_size"] == 227
    assert cfg["split"] == {"mode": "patient", "val_fraction": 0.2}
    assert cfg["model"] == {"pretrained": True, "num_classes": 2}
    train = cfg["train"]
    assert train["lr"] == pytest.approx(1e-3)
    assert train["momentum"] == pytest.approx(0.9)
    assert train["weight_decay"] == pytest.approx(1e-4)
    assert (train["batch_size"], train["epochs"]) == (16, 15)
    assert train["use_class_weights"] is True
