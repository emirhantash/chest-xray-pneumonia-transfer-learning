"""Reproducibility, configuration, and device helpers."""

import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> torch.Generator:
    """Seed every random number generator used in the project.

    Returns a seeded ``torch.Generator`` to pass to DataLoaders so that shuffling
    is reproducible as well.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels (slightly slower). Note: MPS has no equivalent
    # switch, so runs on Apple Silicon may not be bit-exact.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive numpy/random seeds from the torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(path: str) -> dict:
    """Load a YAML configuration file into a nested dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_revision() -> dict:
    """Commit hash of the repository and whether it has uncommitted changes.

    Stored with runs and HPO studies for traceability; ``commit`` is None when git or the
    repository is unavailable (e.g. a copied source tree).
    """
    repo = Path(__file__).resolve().parents[1]

    def git(*args):
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=no")
    except OSError:  # git is not installed
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def get_device() -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
