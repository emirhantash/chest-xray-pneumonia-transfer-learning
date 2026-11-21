"""Data pipeline: file listing, patient IDs, stratified split, transforms, loaders."""

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils import seed_worker

# Class index convention: 0 = NORMAL, 1 = PNEUMONIA (positive class).
CLASSES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = len(CLASSES)
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Patient ID patterns inferred from the dataset's file names:
#   person1000_bacteria_2931.jpeg -> person1000
#   IM-0115-0001.jpeg             -> IM-0115
#   NORMAL2-IM-0383-0001.jpeg     -> NORMAL2-IM-0383
_PATIENT_PATTERNS = [
    re.compile(r"^(person\d+)_"),
    re.compile(r"^((?:NORMAL2-)?IM-\d+)-\d+"),
]


def list_images(split_dir):
    """Return sorted image paths and integer labels from ``split_dir/<CLASS>/``."""
    split_dir = Path(split_dir)
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {cls_dir}. 'data.root' must point to the folder "
                "that directly contains train/ and test/."
            )
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."):
                paths.append(p)
                labels.append(label)
    return paths, labels


def patient_id(path) -> str:
    """Parse the patient ID from a file name; fall back to the file name itself."""
    name = Path(path).name
    for pattern in _PATIENT_PATTERNS:
        match = pattern.match(name)
        if match:
            return match.group(1)
    return name


def split_train_val(paths, labels, val_fraction, mode, seed):
    """Stratified train/val split of the original train set.

    mode="patient": all images of a patient go to the same subset
        (StratifiedGroupKFold with n_splits = round(1 / val_fraction); the first fold is
        the validation set, so the val share is approximately 1 / n_splits).
    mode="image": plain image-level stratified split (interim report setting).

    Returns sorted index arrays (train_idx, val_idx).
    """
    indices = np.arange(len(paths))
    labels = np.asarray(labels)
    if mode == "patient":
        groups = [patient_id(p) for p in paths]
        n_splits = round(1 / val_fraction)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(sgkf.split(indices, labels, groups))
    elif mode == "image":
        train_idx, val_idx = train_test_split(
            indices, test_size=val_fraction, stratify=labels, random_state=seed
        )
    else:
        raise ValueError(f"Unknown split mode: {mode!r} (expected 'patient' or 'image')")
    return np.sort(train_idx), np.sort(val_idx)


def build_splits(cfg) -> dict:
    """Return {"train": (paths, labels), "val": (...), "test": (...)}.

    train/val come from the original train folder; test is the untouched original test
    folder. The original val folder (16 images) is not used.
    """
    root = Path(cfg["data"]["root"])
    paths, labels = list_images(root / "train")
    train_idx, val_idx = split_train_val(
        paths, labels, cfg["split"]["val_fraction"], cfg["split"]["mode"], cfg["seed"]
    )
    return {
        "train": ([paths[i] for i in train_idx], [labels[i] for i in train_idx]),
        "val": ([paths[i] for i in val_idx], [labels[i] for i in val_idx]),
        "test": list_images(root / "test"),
    }


def compute_class_weights(labels, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency weights n / (K * n_c), for CrossEntropyLoss(weight=...)."""
    counts = np.bincount(labels, minlength=num_classes)
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def build_transform(image_size: int):
    """Grayscale -> resize -> replicate to 3 channels -> tensor -> ImageNet normalization."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ChestXrayDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            img = img.convert("L")  # a few files are stored as RGB
        return self.transform(img), self.labels[idx]


def build_dataloaders(splits, cfg, generator) -> dict:
    """Create train/val/test DataLoaders. Only the training loader is shuffled."""
    transform = build_transform(cfg["data"]["image_size"])
    loaders = {}
    for name, (paths, labels) in splits.items():
        loaders[name] = DataLoader(
            ChestXrayDataset(paths, labels, transform),
            batch_size=cfg["train"]["batch_size"],
            shuffle=(name == "train"),
            num_workers=cfg["data"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
        )
    return loaders
