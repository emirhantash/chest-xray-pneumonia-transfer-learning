"""Phase 1 sanity checks: dataset counts, train/val split, class weights,
patient overlap, sample images, and one forward pass through the model.

Usage (from the repo root):
    python scripts/check_data.py --config configs/baseline.yaml [--data-root PATH] [--no-pretrained]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch

from src.data import (
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_dataloaders,
    build_splits,
    compute_class_weights,
    patient_id,
)
from src.model import build_alexnet
from src.utils import get_device, load_config, set_seed

# Counts of the original Kaggle folders.
EXPECTED_COUNTS = {
    "train": {"NORMAL": 1341, "PNEUMONIA": 3875},
    "test": {"NORMAL": 234, "PNEUMONIA": 390},
}


def class_counts(labels):
    counts = np.bincount(labels, minlength=len(CLASSES))
    return {cls: int(counts[i]) for i, cls in enumerate(CLASSES)}


def save_sample_grid(images, labels, path, n=8):
    """Undo the normalization and save the first n images of a batch."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    n = min(n, len(images))
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6))
    # The grid can have one more axis than images (odd n); extra axes are hidden below.
    for ax, img, label in zip(axes.flat, images[:n], labels[:n], strict=False):
        img = (img * std + mean).clamp(0, 1)
        ax.imshow(img[0], cmap="gray")  # channels are identical copies
        ax.set_title(CLASSES[int(label)])
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data-root", help="Override data.root from the config")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip downloading ImageNet weights (offline check)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    cfg["data"]["cache"] = False  # only one batch is loaded here
    generator = set_seed(cfg["seed"])
    out_dir = Path(cfg["output_dir"]) / "check_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    # 1. Original folder counts vs. expected values
    print(f"Data root: {cfg['data']['root']}")
    splits = build_splits(cfg)
    original = {
        "train": class_counts(splits["train"][1] + splits["val"][1]),
        "test": class_counts(splits["test"][1]),
    }
    print("\n[1] Original folder counts (found / expected)")
    for folder, counts in original.items():
        for cls in CLASSES:
            found, expected = counts[cls], EXPECTED_COUNTS[folder][cls]
            status = "OK" if found == expected else "MISMATCH"
            print(f"  {folder:5s} {cls:9s} {found:5d} / {expected:5d}  {status}")
            if found != expected:
                failures.append(f"{folder}/{cls} count {found} != {expected}")

    # 2. Split statistics
    mode, seed = cfg["split"]["mode"], cfg["seed"]
    print(f"\n[2] Split (mode={mode}, val_fraction={cfg['split']['val_fraction']}, seed={seed})")
    if cfg["split"]["file"]:
        # The split file is authoritative; recomputing only shows whether this
        # scikit-learn version would reproduce it (see D9).
        computed = build_splits({**cfg, "split": {**cfg["split"], "file": None}})
        same = computed["val"][0] == splits["val"][0]
        print(f"  source: split file {cfg['split']['file']}")
        print(
            f"  recomputed with scikit-learn {sklearn.__version__}: "
            + ("identical" if same else "DIFFERENT (the split file is used)")
        )
    else:
        print(f"  source: computed with scikit-learn {sklearn.__version__}")
    n_total = len(splits["train"][1]) + len(splits["val"][1])
    print(
        f"  {'subset':6s} {'NORMAL':>7s} {'PNEUM.':>7s} {'total':>6s} {'share':>6s} "
        f"{'%pneum':>7s} {'patients':>9s}"
    )
    for name, (paths, labels) in splits.items():
        c = class_counts(labels)
        share = f"{len(labels) / n_total:6.1%}" if name != "test" else "     -"
        n_patients = len({patient_id(p) for p in paths})
        print(
            f"  {name:6s} {c['NORMAL']:7d} {c['PNEUMONIA']:7d} {len(labels):6d} {share} "
            f"{c['PNEUMONIA'] / len(labels):7.1%} {n_patients:9d}"
        )

    # 3. Class weights (training subset only)
    weights = compute_class_weights(splits["train"][1])
    print("\n[3] Class weights (train subset, n / (K * n_c))")
    for cls, w in zip(CLASSES, weights.tolist(), strict=True):
        print(f"  {cls:9s} {w:.4f}")

    # 4. Patient overlap diagnostics
    # patient_id raises for an unparsable name, so every file has an ID at this point.
    ids = {name: {patient_id(p) for p in paths} for name, (paths, _) in splits.items()}
    train_val = len(ids["train"] & ids["val"])
    trainval_test = len((ids["train"] | ids["val"]) & ids["test"])
    print("\n[4] Patient-ID overlap (IDs parsed from file names)")
    print("  files without a parsable patient ID: 0 (parsing is strict)")
    print(f"  train & val shared patient IDs:      {train_val}")
    print(
        f"  train/val & test shared patient IDs: {trainval_test} "
        "(numbering may restart per folder; not verified as the same patients)"
    )
    if mode == "patient" and train_val > 0:
        failures.append(f"patient split has {train_val} patients in both train and val")

    split_csv = out_dir / f"split_{mode}_seed{seed}.csv"
    root = Path(cfg["data"]["root"])
    with open(split_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "path", "label", "class", "patient_id"])
        for name, (paths, labels) in splits.items():
            for p, y in zip(paths, labels, strict=True):
                writer.writerow(
                    [name, p.relative_to(root).as_posix(), y, CLASSES[y], patient_id(p)]
                )
    print(f"  split file list saved to {split_csv}")

    # 5. One batch: tensor stats and sample images
    loaders = build_dataloaders(splits, cfg, generator)
    images, labels = next(iter(loaders["train"]))
    print("\n[5] One training batch")
    print(
        f"  images {tuple(images.shape)} {images.dtype}, "
        f"min {images.min():.3f}, max {images.max():.3f}, mean {images.mean():.3f}"
    )
    print(f"  labels {labels.tolist()}")
    size = cfg["data"]["image_size"]
    if tuple(images.shape[1:]) != (3, size, size):
        failures.append(f"unexpected image shape {tuple(images.shape)}")
    grid_path = out_dir / "samples.png"
    save_sample_grid(images, labels, grid_path)
    print(f"  sample grid saved to {grid_path}")

    # 6. Forward pass through the model
    device = get_device()
    pretrained = cfg["model"]["pretrained"] and not args.no_pretrained
    model = build_alexnet(cfg["model"]["num_classes"], pretrained=pretrained).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
    expected_shape = (images.shape[0], cfg["model"]["num_classes"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[6] Model forward pass (device={device}, pretrained={pretrained})")
    print(f"  parameters: {n_params:,}")
    print(f"  logits shape {tuple(logits.shape)}, expected {expected_shape}")
    if tuple(logits.shape) != expected_shape:
        failures.append(f"logits shape {tuple(logits.shape)} != {expected_shape}")

    print("\n" + ("ALL CHECKS PASSED" if not failures else "CHECKS FAILED:"))
    for msg in failures:
        print(f"  - {msg}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
