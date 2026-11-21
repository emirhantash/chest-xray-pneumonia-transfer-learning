"""Shared fixtures: a tiny synthetic dataset that mimics the Kaggle folder layout.

The file names follow the real naming conventions so that patient-ID parsing and the
patient-level split are exercised the same way as on the real data. No real images
and no network access are needed.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"

# (file name, color mode). Several patients have more than one image.
TRAIN_NORMAL = (
    [(f"IM-{pid:04d}-{k:04d}.jpeg", "L") for pid in range(1, 7) for k in range(1, 3)]
    + [(f"IM-{pid:04d}-0001.jpeg", "L") for pid in range(7, 11)]
    + [(f"NORMAL2-IM-{pid:04d}-{k:04d}.jpeg", "L") for pid in range(1, 5) for k in range(1, 3)]
    # Same numeric ID as IM-0001 but a different namespace (a different patient).
    + [("NORMAL2-IM-0001-0003.jpeg", "L")]
)
TRAIN_PNEUMONIA = (
    [(f"person{pid}_bacteria_{pid * 10 + k}.jpeg", "L") for pid in range(1, 11) for k in range(3)]
    + [(f"person{pid}_virus_{pid * 10 + k}.jpeg", "L") for pid in range(11, 21) for k in range(2)]
    # A few RGB files, as in the real train/PNEUMONIA folder.
    + [(f"person{pid}_bacteria_{pid * 10 + 5}.jpeg", "RGB") for pid in range(21, 25)]
)
TEST_NORMAL = [(f"IM-{pid:04d}-0001.jpeg", "L") for pid in range(101, 104)]
TEST_PNEUMONIA = [(f"person{pid}_virus_{pid}.jpeg", "L") for pid in range(101, 105)]


def _write_image(path: Path, mode: str, rng: np.random.Generator) -> None:
    """Write a small random image with a non-square size (like the real data)."""
    height, width = rng.integers(20, 40), rng.integers(30, 60)
    if mode == "RGB":
        array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    else:
        array = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    Image.fromarray(array).save(path)


@pytest.fixture(scope="session")
def data_root(tmp_path_factory) -> Path:
    """Synthetic ``<root>/{train,test}/{NORMAL,PNEUMONIA}/`` folder tree."""
    root = tmp_path_factory.mktemp("chest_xray")
    rng = np.random.default_rng(0)
    layout = {
        "train/NORMAL": TRAIN_NORMAL,
        "train/PNEUMONIA": TRAIN_PNEUMONIA,
        "test/NORMAL": TEST_NORMAL,
        "test/PNEUMONIA": TEST_PNEUMONIA,
    }
    for folder, files in layout.items():
        folder_path = root / folder
        folder_path.mkdir(parents=True)
        for name, mode in files:
            _write_image(folder_path / name, mode, rng)
    # Files that must be ignored: a macOS AppleDouble file and a non-image file.
    (root / "train/PNEUMONIA/._person1_bacteria_10.jpeg").write_bytes(b"\x00\x05\x16\x07")
    (root / "train/NORMAL/notes.txt").write_text("not an image")
    return root


@pytest.fixture()
def cfg(data_root) -> dict:
    """The baseline config pointed at the synthetic data, with test-friendly loaders."""
    config = load_config(BASELINE_CONFIG)
    config["data"]["root"] = str(data_root)
    config["data"]["num_workers"] = 0
    config["train"]["batch_size"] = 4
    return config
