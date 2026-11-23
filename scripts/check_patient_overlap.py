"""Check whether train and test images that share a parsed patient ID show the same patient (D8).

Usage (from the repo root):
    python scripts/check_patient_overlap.py --data-root PATH [--size 32]

Patient IDs are parsed from file names, and some IDs occur in both the original train
and test folders. This script looks at image content only (no model, no labels used for
anything but grouping):

1. exact duplicate files (MD5) and near-duplicate images between train and test;
2. a nearest-neighbour test: within train, how often is the most similar other image
   (Pearson correlation of small grayscale thumbnails) from the same patient? This is
   the positive control. If shared IDs were the same patients, test images with a shared
   ID should have a train image of that ID as their nearest neighbour at a similar rate.
   The p-value is the binomial probability of observing at most the test count if the
   test rate equalled the train rate.

Writes outputs/check_patient_overlap/report.json.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import list_images, patient_id
from src.utils import load_config

NEAR_DUPLICATE_CORR = 0.99


def _thumbnail(path, size: int):
    raw = Path(path).read_bytes()
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32).ravel()
    return hashlib.md5(raw).hexdigest(), (x - x.mean()) / (x.std() + 1e-6)


def _load(split_dir: Path, size: int):
    paths, labels = list_images(split_dir)
    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(lambda p: _thumbnail(p, size), paths))
    ids = np.array([patient_id(p) for p in paths])
    return ids, np.asarray(labels), {h for h, _ in out}, np.stack([x for _, x in out])


def _binom_cdf(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data-root", help="Override data.root from the config")
    parser.add_argument("--size", type=int, default=32, help="Thumbnail side in pixels")
    parser.add_argument("--out", default="outputs/check_patient_overlap")
    args = parser.parse_args()
    root = Path(args.data_root or load_config(args.config)["data"]["root"])

    tr_ids, tr_y, tr_hash, tr_x = _load(root / "train", args.size)
    te_ids, te_y, te_hash, te_x = _load(root / "test", args.size)
    n_pix = tr_x.shape[1]
    test_vs_train = te_x @ tr_x.T / n_pix  # Pearson correlation of the thumbnails

    # Positive control: nearest other train image of train images of multi-image patients.
    members = defaultdict(list)
    for i, key in enumerate(zip(tr_ids, tr_y, strict=True)):
        members[key].append(i)
    multi = np.array([i for idx in members.values() if len(idx) > 1 for i in idx])
    hits_train = 0
    for start in range(0, len(multi), 512):
        rows = multi[start : start + 512]
        sim = tr_x[rows] @ tr_x.T / n_pix
        sim[np.arange(len(rows)), rows] = -np.inf
        hits_train += int(np.sum(tr_ids[sim.argmax(1)] == tr_ids[rows]))
    rate_train = hits_train / len(multi)

    shared = np.array([j for j, key in enumerate(zip(te_ids, te_y, strict=True)) if key in members])
    nearest = test_vs_train.argmax(1)
    hits_test = int(np.sum(tr_ids[nearest[shared]] == te_ids[shared])) if shared.size else 0

    report = {
        "thumbnail_size": args.size,
        "exact_duplicate_files": len(tr_hash & te_hash),
        "near_duplicate_test_images": int(np.sum(test_vs_train.max(1) > NEAR_DUPLICATE_CORR)),
        "near_duplicate_threshold": NEAR_DUPLICATE_CORR,
        "shared_patient_ids": len({(te_ids[j], te_y[j]) for j in shared}),
        "test_images_with_shared_id": int(shared.size),
        "train_nn_same_patient": [hits_train, int(len(multi))],
        "train_nn_same_patient_rate": rate_train,
        "test_nn_same_id": [hits_test, int(shared.size)],
        "expected_test_nn_same_id_if_same_patients": rate_train * shared.size,
        "p_value_at_most_observed": _binom_cdf(hits_test, int(shared.size), rate_train),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, value in report.items():
        print(f"{key:44s} {value}")
    print(f"Report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
