"""Data pipeline: file listing, patient IDs, stratified split, transforms, loaders."""

import csv
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
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
SPLIT_FILE_FIELDS = ["subset", "path", "label", "patient_id"]

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
    """Parse the patient ID from a file name.

    Raises ``ValueError`` for a name that matches no known convention: silently treating
    such a file as its own patient could put images of one patient on both sides of the
    patient-level split.
    """
    name = Path(path).name
    for pattern in _PATIENT_PATTERNS:
        match = pattern.match(name)
        if match:
            return match.group(1)
    raise ValueError(
        f"No patient ID in file name {name!r}; expected person<ID>_..., IM-<ID>-... or "
        "NORMAL2-IM-<ID>-..."
    )


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


def write_split_file(path, splits, root) -> None:
    """Save the train/val assignment as a CSV of paths relative to ``root`` (no images)."""
    path, root = Path(path), Path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")  # same bytes on every OS
        writer.writerow(SPLIT_FILE_FIELDS)
        for subset in ("train", "val"):
            for p, y in zip(*splits[subset], strict=True):
                writer.writerow([subset, p.relative_to(root).as_posix(), y, patient_id(p)])


def read_split_file(path) -> dict:
    """Read a split file into ``{relative path: (subset, label)}``."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != SPLIT_FILE_FIELDS:
            raise ValueError(
                f"{path}: expected columns {SPLIT_FILE_FIELDS}, got {reader.fieldnames}"
            )
        entries = {row["path"]: (row["subset"], int(row["label"])) for row in reader}
    unknown = {subset for subset, _ in entries.values()} - {"train", "val"}
    if unknown:
        raise ValueError(f"{path}: unknown subsets {sorted(unknown)}")
    return entries


def _splits_from_file(path, root, paths, labels):
    """Assign the listed train-folder images to train/val as recorded in a split file.

    The file must cover exactly the images in the train folder, with matching labels, so
    a different dataset copy or a stale file fails loudly instead of changing the split.
    """
    entries = read_split_file(path)
    listed = {p.relative_to(root).as_posix(): (p, y) for p, y in zip(paths, labels, strict=True)}
    missing, extra = entries.keys() - listed.keys(), listed.keys() - entries.keys()
    if missing or extra:
        raise ValueError(
            f"Split file {path} does not match {root / 'train'}: {len(missing)} listed "
            f"files not found, {len(extra)} images not in the file (e.g. "
            f"{sorted(missing or extra)[0]})."
        )
    splits = {"train": ([], []), "val": ([], [])}
    for rel, (p, y) in listed.items():  # listing order, as in the computed split
        subset, file_label = entries[rel]
        if file_label != y:
            raise ValueError(f"Split file {path}: label {file_label} != folder label {y} ({rel})")
        splits[subset][0].append(p)
        splits[subset][1].append(y)
    return splits


def build_splits(cfg, include_test: bool = True) -> dict:
    """Return {"train": (paths, labels), "val": (...), "test": (...)}.

    train/val come from the original train folder: read from ``split.file`` if set,
    otherwise computed with :func:`split_train_val`. test is the untouched original test
    folder and is only listed when ``include_test`` is true (training never needs it).
    The original val folder (16 images) is not used.
    """
    root = Path(cfg["data"]["root"])
    paths, labels = list_images(root / "train")
    if cfg["split"]["file"]:
        splits = _splits_from_file(cfg["split"]["file"], root, paths, labels)
    else:
        train_idx, val_idx = split_train_val(
            paths, labels, cfg["split"]["val_fraction"], cfg["split"]["mode"], cfg["seed"]
        )
        splits = {
            "train": ([paths[i] for i in train_idx], [labels[i] for i in train_idx]),
            "val": ([paths[i] for i in val_idx], [labels[i] for i in val_idx]),
        }
    if include_test:
        splits["test"] = list_images(root / "test")
    return splits


def split_fingerprint(splits, root) -> str:
    """Short hash of the train/val assignment (relative paths and labels).

    Stored in checkpoints so that a resumed run or a later evaluation can verify that it
    uses exactly the same split, independent of where the data lives.
    """
    digest = hashlib.sha256()
    for subset in ("train", "val"):
        for p, y in zip(*splits[subset], strict=True):
            line = f"{subset},{Path(p).relative_to(root).as_posix()},{y}\n"
            digest.update(line.encode())
    return digest.hexdigest()[:16]


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


def load_gray(path) -> Image.Image:
    """Open an image as single-channel grayscale (a few files are stored as RGB)."""
    with Image.open(path) as img:
        return img.convert("L")


def preload_images(paths, image_size: int) -> torch.Tensor:
    """Decode and resize all images once; returns a uint8 tensor (N, H, W) in shared memory.

    Uses the same Resize as :func:`build_transform`, so the cached and the on-the-fly
    paths give identical tensors (the second Resize is a no-op at the target size).
    Shared memory lets DataLoader workers read the cache without copying it.
    """
    resize = transforms.Resize((image_size, image_size))

    def load(path):
        return np.asarray(resize(load_gray(path)), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        arrays = list(pool.map(load, paths))
    return torch.from_numpy(np.stack(arrays)).share_memory_()


class ChestXrayDataset(Dataset):
    """Images and labels; optionally served from an in-memory cache (``preload_size``)."""

    def __init__(self, paths, labels, transform, preload_size: int | None = None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.cache = preload_images(paths, preload_size) if preload_size else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.cache is not None:
            img = Image.fromarray(self.cache[idx].numpy())
        else:
            img = load_gray(self.paths[idx])
        return self.transform(img), self.labels[idx]


def build_datasets(splits, cfg) -> dict:
    """Create one dataset per split.

    With ``data.cache`` the resized grayscale images are kept in RAM (about 270 MB for
    train + val at 227 x 227); without augmentation this gives identical batches. The
    datasets can be reused for loaders with different batch sizes (HPO trials).
    """
    image_size = cfg["data"]["image_size"]
    transform = build_transform(image_size)
    preload_size = image_size if cfg["data"]["cache"] else None
    return {
        name: ChestXrayDataset(paths, labels, transform, preload_size)
        for name, (paths, labels) in splits.items()
    }


def make_loader(dataset, shuffle: bool, cfg, generator) -> DataLoader:
    """DataLoader with the batch size and worker settings of ``cfg``."""
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_dataloaders(splits, cfg, generator) -> dict:
    """Create one DataLoader per split. Only the training loader is shuffled."""
    datasets = build_datasets(splits, cfg)
    return {
        name: make_loader(dataset, name == "train", cfg, generator)
        for name, dataset in datasets.items()
    }
