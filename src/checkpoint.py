"""Checkpoint saving/loading and random-number-generator state, for exact resume."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(state: dict, path) -> None:
    """Write ``state`` atomically: a crash or a lost notebook session during the write
    leaves the previous checkpoint intact instead of a truncated file."""
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, map_location="cpu") -> dict:
    """Load a checkpoint written by :func:`save_checkpoint`.

    ``weights_only=False`` is needed because the file also stores Python/NumPy RNG states
    and the config. Only load checkpoints produced by this project.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def get_rng_state() -> dict:
    """Capture the Python, NumPy, PyTorch CPU and CUDA generator states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict) -> None:
    """Restore the states captured by :func:`get_rng_state`.

    The CUDA state is restored only when CUDA is available and the device count matches
    (resuming on a different machine type is supported but not bit-exact).
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state["cuda"]
    if cuda_state is not None and torch.cuda.is_available():
        if len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_state)
