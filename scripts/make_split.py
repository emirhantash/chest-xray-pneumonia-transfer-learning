"""Compute the train/val split from the config and save it as a split file.

Usage (from the repo root):
    python scripts/make_split.py --config configs/baseline.yaml [--data-root PATH]
        [--out splits/patient_seed42.csv] [--force]

The split is computed from split.mode / split.val_fraction / seed (split.file is
ignored) and written as relative paths, labels, and patient IDs; no images. Training
then reads this file (split.file), so the split no longer depends on the scikit-learn
version.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sklearn

from src.data import build_splits, patient_id, write_split_file
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data-root", help="Override data.root from the config")
    parser.add_argument("--out", help="Output CSV (default: splits/<mode>_seed<seed>.csv)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    mode, seed = cfg["split"]["mode"], cfg["seed"]
    out = Path(args.out or f"splits/{mode}_seed{seed}.csv")
    if out.exists() and not args.force:
        parser.error(f"{out} exists; pass --force to overwrite it.")

    cfg["split"]["file"] = None  # always compute
    splits = build_splits(cfg, include_test=False)
    write_split_file(out, splits, cfg["data"]["root"])

    print(f"Split (mode={mode}, seed={seed}, scikit-learn {sklearn.__version__}) -> {out}")
    for name, (paths, labels) in splits.items():
        n_patients = len({patient_id(p) for p in paths})
        print(
            f"  {name:5s} {len(labels):5d} images ({labels.count(0)} NORMAL / "
            f"{labels.count(1)} PNEUMONIA), {n_patients} patients"
        )


if __name__ == "__main__":
    main()
