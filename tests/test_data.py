"""Tests for the data pipeline in src/data.py (synthetic data only)."""

import math

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import (
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ChestXrayDataset,
    build_dataloaders,
    build_splits,
    build_transform,
    compute_class_weights,
    list_images,
    patient_id,
    read_split_file,
    split_fingerprint,
    split_train_val,
    write_split_file,
)
from src.utils import set_seed
from tests.conftest import TEST_NORMAL, TEST_PNEUMONIA, TRAIN_NORMAL, TRAIN_PNEUMONIA

N_TRAIN = len(TRAIN_NORMAL) + len(TRAIN_PNEUMONIA)


# --- patient IDs -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("person1000_bacteria_2931.jpeg", "person1000"),
        ("person1_virus_6.jpeg", "person1"),
        ("IM-0115-0001.jpeg", "IM-0115"),
        ("NORMAL2-IM-0383-0001.jpeg", "NORMAL2-IM-0383"),
        ("unknown_file.jpeg", "unknown_file.jpeg"),  # fallback: the file name itself
    ],
)
def test_patient_id(name, expected):
    assert patient_id(name) == expected


def test_patient_id_accepts_paths(data_root):
    path = data_root / "train" / "NORMAL" / "IM-0001-0002.jpeg"
    assert patient_id(path) == "IM-0001"


def test_im_and_normal2_namespaces_are_distinct():
    assert patient_id("IM-0001-0001.jpeg") != patient_id("NORMAL2-IM-0001-0001.jpeg")


# --- file listing ------------------------------------------------------------------------


def test_list_images_counts_labels_and_skips_non_images(data_root):
    paths, labels = list_images(data_root / "train")
    names = {p.name for p in paths}
    assert len(paths) == N_TRAIN
    assert labels.count(0) == len(TRAIN_NORMAL)
    assert labels.count(1) == len(TRAIN_PNEUMONIA)
    assert "._person1_bacteria_10.jpeg" not in names
    assert "notes.txt" not in names
    # Label order follows CLASSES: all NORMAL first, then all PNEUMONIA.
    assert labels == sorted(labels)
    assert all(p.parent.name == CLASSES[y] for p, y in zip(paths, labels, strict=True))


def test_list_images_missing_class_folder_raises(tmp_path):
    (tmp_path / "NORMAL").mkdir()
    with pytest.raises(FileNotFoundError, match="PNEUMONIA"):
        list_images(tmp_path)


# --- train/val split ---------------------------------------------------------------------


def _train_listing(data_root):
    paths, labels = list_images(data_root / "train")
    return paths, np.asarray(labels)


def test_patient_split_has_no_shared_patients(data_root):
    paths, labels = _train_listing(data_root)
    train_idx, val_idx = split_train_val(paths, labels, 0.2, "patient", seed=42)
    train_ids = {patient_id(paths[i]) for i in train_idx}
    val_ids = {patient_id(paths[i]) for i in val_idx}
    assert train_ids.isdisjoint(val_ids)


def test_patient_split_is_a_partition(data_root):
    paths, labels = _train_listing(data_root)
    train_idx, val_idx = split_train_val(paths, labels, 0.2, "patient", seed=42)
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert sorted(np.concatenate([train_idx, val_idx]).tolist()) == list(range(len(paths)))
    # Returned indices are sorted.
    assert np.all(np.diff(train_idx) > 0) and np.all(np.diff(val_idx) > 0)


def test_patient_split_is_stratified(data_root):
    paths, labels = _train_listing(data_root)
    train_idx, val_idx = split_train_val(paths, labels, 0.2, "patient", seed=42)
    # Both classes in both subsets, a roughly 20 % validation share, and a class ratio
    # close to the overall one (tolerances are loose: there are only ~40 patients).
    assert set(labels[train_idx]) == {0, 1}
    assert set(labels[val_idx]) == {0, 1}
    assert 0.1 <= len(val_idx) / len(paths) <= 0.3
    assert abs(labels[val_idx].mean() - labels.mean()) < 0.15


def test_split_is_reproducible_with_the_same_seed(data_root):
    paths, labels = _train_listing(data_root)
    for mode in ("patient", "image"):
        first = split_train_val(paths, labels, 0.2, mode, seed=42)
        second = split_train_val(paths, labels, 0.2, mode, seed=42)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


def test_split_depends_on_the_seed(data_root):
    paths, labels = _train_listing(data_root)
    val_sets = {tuple(split_train_val(paths, labels, 0.2, "patient", seed=s)[1]) for s in range(5)}
    assert len(val_sets) > 1


def test_image_split(data_root):
    paths, labels = _train_listing(data_root)
    train_idx, val_idx = split_train_val(paths, labels, 0.2, "image", seed=42)
    assert len(val_idx) == math.ceil(0.2 * len(paths))  # sklearn rounds the test size up
    assert len(train_idx) + len(val_idx) == len(paths)
    assert set(labels[val_idx]) == {0, 1}


def test_unknown_split_mode_raises(data_root):
    paths, labels = _train_listing(data_root)
    with pytest.raises(ValueError, match="Unknown split mode"):
        split_train_val(paths, labels, 0.2, "random", seed=42)


def test_build_splits_can_skip_the_test_set(cfg):
    assert set(build_splits(cfg, include_test=False)) == {"train", "val"}


# --- split file --------------------------------------------------------------------------


def test_split_file_roundtrip(cfg, data_root, tmp_path):
    computed = build_splits(cfg, include_test=False)
    path = tmp_path / "split.csv"
    write_split_file(path, computed, data_root)

    cfg["split"]["file"] = str(path)
    from_file = build_splits(cfg, include_test=False)
    assert from_file == computed  # same images, labels, and order
    assert split_fingerprint(from_file, data_root) == split_fingerprint(computed, data_root)

    entries = read_split_file(path)
    assert len(entries) == N_TRAIN
    assert all(rel.startswith("train/") for rel in entries)  # relative, POSIX-style paths


def test_split_file_does_not_depend_on_split_settings(cfg, data_root, tmp_path):
    path = tmp_path / "split.csv"
    write_split_file(path, build_splits(cfg, include_test=False), data_root)
    expected = build_splits(cfg, include_test=False)

    # Once a split file is given, seed / mode no longer change the split.
    cfg["split"].update(file=str(path), mode="image")
    cfg["seed"] = 7
    assert build_splits(cfg, include_test=False) == expected


def _rewrite_split_file(path, transform):
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(transform(lines)) + "\n", encoding="utf-8")


def test_split_file_must_cover_the_train_folder(cfg, data_root, tmp_path):
    path = tmp_path / "split.csv"
    write_split_file(path, build_splits(cfg, include_test=False), data_root)
    _rewrite_split_file(path, lambda lines: lines[:-1])  # drop one image
    cfg["split"]["file"] = str(path)
    with pytest.raises(ValueError, match="1 images not in the file"):
        build_splits(cfg)


def test_split_file_label_mismatch_raises(cfg, data_root, tmp_path):
    path = tmp_path / "split.csv"
    write_split_file(path, build_splits(cfg, include_test=False), data_root)

    def flip_first_label(lines):
        subset, rel, label, pid = lines[1].split(",")
        return [lines[0], f"{subset},{rel},{1 - int(label)},{pid}", *lines[2:]]

    _rewrite_split_file(path, flip_first_label)
    cfg["split"]["file"] = str(path)
    with pytest.raises(ValueError, match="folder label"):
        build_splits(cfg)


def test_split_fingerprint_changes_with_the_split(cfg, data_root):
    splits = build_splits(cfg, include_test=False)
    other = build_splits({**cfg, "seed": cfg["seed"] + 1}, include_test=False)
    assert splits["val"] != other["val"]
    assert split_fingerprint(splits, data_root) != split_fingerprint(other, data_root)


def test_build_splits_keeps_test_set_untouched(cfg, data_root):
    splits = build_splits(cfg)
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"][0]) + len(splits["val"][0]) == N_TRAIN
    test_paths, test_labels = splits["test"]
    assert len(test_paths) == len(TEST_NORMAL) + len(TEST_PNEUMONIA)
    assert all(p.parent.parent == data_root / "test" for p in test_paths)
    assert test_labels.count(1) == len(TEST_PNEUMONIA)
    # No training/validation image comes from the test folder.
    for name in ("train", "val"):
        assert all(p.parent.parent == data_root / "train" for p in splits[name][0])


# --- class weights -----------------------------------------------------------------------


def test_class_weights_formula():
    weights = compute_class_weights([0, 0, 0, 1])
    assert weights.dtype == torch.float32
    # n / (K * n_c) = 4 / (2 * 3) and 4 / (2 * 1)
    torch.testing.assert_close(weights, torch.tensor([4 / 6, 2.0]))


def test_class_weights_documented_values():
    # Class counts of the documented patient-level training subset (seed 42).
    labels = [0] * 1072 + [1] * 3100
    weights = compute_class_weights(labels)
    torch.testing.assert_close(weights, torch.tensor([1.9459, 0.6729]), atol=1e-4, rtol=0)


def test_class_weights_balance_classes():
    labels = [0] * 7 + [1] * 21
    weights = compute_class_weights(labels)
    counts = torch.tensor([7.0, 21.0])
    # Every class contributes the same total weight n / K.
    torch.testing.assert_close(weights * counts, torch.full((2,), len(labels) / 2))


# --- transforms and dataset --------------------------------------------------------------


def _unnormalize(tensor):
    """Undo the per-channel ImageNet normalization (channels differ only after it)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


def _assert_identical_channels(tensor):
    pixels = _unnormalize(tensor)
    torch.testing.assert_close(pixels[0], pixels[1])
    torch.testing.assert_close(pixels[0], pixels[2])


def test_transform_output_shape_and_channels():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (30, 50), np.uint8))
    tensor = build_transform(64)(image)
    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32
    _assert_identical_channels(tensor)
    pixels = _unnormalize(tensor)
    assert pixels.min() >= -1e-5 and pixels.max() <= 1 + 1e-5


def test_dataset_converts_rgb_to_gray(tmp_path):
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 200, 100, 30  # clearly different channels
    path = tmp_path / "person1_bacteria_1.png"
    Image.fromarray(rgb).save(path)

    transform = build_transform(32)
    image, label = ChestXrayDataset([path], [1], transform)[0]
    assert label == 1
    assert image.shape == (3, 32, 32)
    _assert_identical_channels(image)
    expected = transform(Image.fromarray(rgb).convert("L"))
    torch.testing.assert_close(image, expected)


def test_rgb_and_grayscale_copies_give_identical_tensors(tmp_path):
    # The same picture stored as 1-channel and as 3-channel must be indistinguishable,
    # so the storage format cannot act as a shortcut feature.
    gray = np.random.default_rng(1).integers(0, 256, (25, 35), dtype=np.uint8)
    gray_path, rgb_path = tmp_path / "gray.png", tmp_path / "rgb.png"
    Image.fromarray(gray).save(gray_path)
    Image.fromarray(np.stack([gray] * 3, axis=-1)).save(rgb_path)

    dataset = ChestXrayDataset([gray_path, rgb_path], [0, 0], build_transform(32))
    torch.testing.assert_close(dataset[0][0], dataset[1][0], atol=0, rtol=0)


def test_cached_dataset_gives_identical_tensors(data_root):
    paths, labels = list_images(data_root / "train")
    transform = build_transform(48)
    plain = ChestXrayDataset(paths, labels, transform)
    cached = ChestXrayDataset(paths, labels, transform, preload_size=48)
    assert cached.cache.shape == (len(paths), 48, 48)
    assert cached.cache.dtype == torch.uint8
    for i in range(len(paths)):  # includes the RGB files
        image_plain, label_plain = plain[i]
        image_cached, label_cached = cached[i]
        assert label_plain == label_cached
        torch.testing.assert_close(image_plain, image_cached, atol=0, rtol=0)


# --- data loaders ------------------------------------------------------------------------


def _first_train_batch(cfg):
    generator = set_seed(cfg["seed"])
    loaders = build_dataloaders(build_splits(cfg), cfg, generator)
    return loaders, next(iter(loaders["train"]))


def test_dataloaders_batch_shape(cfg):
    loaders, (images, labels) = _first_train_batch(cfg)
    assert set(loaders) == {"train", "val", "test"}
    size = cfg["data"]["image_size"]
    assert images.shape == (cfg["train"]["batch_size"], 3, size, size)
    assert labels.shape == (cfg["train"]["batch_size"],)


def test_dataloader_shuffling_is_reproducible(cfg):
    _, (images_a, labels_a) = _first_train_batch(cfg)
    _, (images_b, labels_b) = _first_train_batch(cfg)
    torch.testing.assert_close(images_a, images_b)
    torch.testing.assert_close(labels_a, labels_b)


def test_cached_and_uncached_loaders_give_identical_batches(cfg):
    _, (images_plain, labels_plain) = _first_train_batch(cfg)
    cfg["data"]["cache"] = True
    _, (images_cached, labels_cached) = _first_train_batch(cfg)
    torch.testing.assert_close(images_plain, images_cached, atol=0, rtol=0)
    torch.testing.assert_close(labels_plain, labels_cached)
