# Pneumonia Detection from Pediatric Chest X-Rays with AlexNet Transfer Learning and Hyperparameter Optimization

[![CI](https://github.com/emirhantash/chest-xray-pneumonia-transfer-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/emirhantash/chest-xray-pneumonia-transfer-learning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

BİM 541 course project (Erciyes University). Binary classification (Normal vs Pneumonia) on
the Kaggle "Chest X-Ray Images (Pneumonia)" dataset with an ImageNet-pretrained AlexNet
(PyTorch), followed by hyperparameter optimization (Bayesian optimization vs. Random Search).

**Status:** The project skeleton, data pipeline, test suite, and CI are done. Training,
metrics, and HPO come next.

## Setup

Requires Python ≥ 3.11 and scikit-learn ≥ 1.8 (the patient-level split depends on the
scikit-learn version, see [D9](docs/deviations_from_interim_report.md#d9-the-patient-level-split-depends-on-the-scikit-learn-version-decided-phase-15)).

```bash
git clone https://github.com/emirhantash/chest-xray-pneumonia-transfer-learning.git
cd chest-xray-pneumonia-transfer-learning
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

# 1) Install PyTorch for your hardware (CUDA / CPU / Apple MPS):
#    https://pytorch.org/get-started/locally/
# 2) Install the remaining dependencies:
pip install -r requirements.txt
```

The code picks the device automatically in this order: CUDA, then Apple MPS, then CPU.

## Data

Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
by Paul Mooney (a version of the data released by Kermany et al.). The data is **not**
included in this repository.

**Option A: Kaggle CLI**

```bash
pip install kaggle
# Put your API token (Kaggle > Settings > API > Create New Token) in
# ~/.kaggle/kaggle.json (Windows: C:\Users\<you>\.kaggle\kaggle.json)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/
# Extract the archive into data/ (e.g. with your OS's unzip tool)
```

**Option B: Manual download.** Download the zip from the Kaggle page and extract it.

The archive contains the same folder twice (`chest_xray/` and `chest_xray/chest_xray/`) plus
a `__MACOSX/` folder. Use either copy and point `data.root` to the folder that **directly**
contains `train/` and `test/`:

```
<data.root>/
├── train/  NORMAL/ (1,341)  PNEUMONIA/ (3,875)
├── test/   NORMAL/ (234)    PNEUMONIA/ (390)
└── val/    (16 images, not used)
```

The default `data.root` is `data/chest_xray` (inside the repo, gitignored). For data stored
elsewhere, edit `configs/baseline.yaml` or pass `--data-root`:

```bash
python scripts/check_data.py --config configs/baseline.yaml --data-root "C:/Users/<you>/Documents/chest_xray"
```

On Kaggle notebooks the path is `/kaggle/input/chest-xray-pneumonia/chest_xray`.

## Data protocol

- **Original test set (624 images)**: used only for the final evaluation, never for
  training, early stopping, or hyperparameter selection.
- **Original val folder (16 images)**: not used (too small to be meaningful).
- **Train / validation**: the original train set (5,216 images) is split ≈80/20, stratified
  by class, with a fixed seed, at the **patient level** (`split.mode: patient`). The patient
  ID is parsed from the file name (`person<ID>_...`, `IM-<ID>-...`,
  `NORMAL2-IM-<ID>-...`), so all images of a patient end up in the same subset.
- **Class imbalance**: inverse-frequency class weights `n / (K * n_c)` computed on the
  training subset are used in `CrossEntropyLoss`.
- **Preprocessing**: grayscale, resized to 227 × 227, replicated to 3 channels, normalized
  with ImageNet mean/std.

## Sanity check

```bash
python scripts/check_data.py --config configs/baseline.yaml
```

This script:
- compares the class counts with the expected values;
- prints the train/val/test sizes, class ratios, patient counts, and class weights;
- reports patient-ID overlap between subsets;
- saves the split file list (`outputs/check_data/split_<mode>_seed<seed>.csv`) and a
  sample image grid (`outputs/check_data/samples.png`);
- passes one batch through AlexNet and checks the output shape.

It prints `ALL CHECKS PASSED` and exits with code 0 on success. Use `--no-pretrained` to
skip the ImageNet weight download (offline).

## Development

Install the development tools on top of the setup above:

```bash
pip install -r requirements-dev.txt
```

Run the tests and the linter from the repository root:

```bash
pytest                  # unit tests on a tiny synthetic dataset (no real data, no network)
ruff check .            # lint
ruff format .           # format (CI runs `ruff format --check .`)
```

The tests build a small fake dataset with the same folder layout and file-name conventions
as the real one, so they run anywhere, including CI. GitHub Actions runs `ruff check`,
`ruff format --check`, and `pytest` with CPU-only PyTorch on Python 3.11 for every push and
pull request to `main`. Tool settings (ruff, pytest) live in `pyproject.toml`.

## Project structure

```
configs/baseline.yaml      # hyperparameters and paths (baseline = interim report, Table 3)
src/utils.py               # set_seed, seed_worker, load_config, get_device
src/data.py                # file listing, patient IDs, split, transforms, loaders, class weights
src/model.py               # build_alexnet
scripts/check_data.py      # Phase 1 sanity checks
tests/                     # pytest suite on synthetic data (fixtures in conftest.py)
docs/deviations_from_interim_report.md   # what changed since the interim report and why
pyproject.toml             # project metadata, ruff and pytest settings
requirements.txt           # runtime dependencies (install PyTorch first)
requirements-dev.txt       # pytest, ruff
.github/workflows/ci.yml   # CI: lint + tests on CPU
```

## Reproducibility

All randomness (Python, NumPy, PyTorch, DataLoader shuffling and workers, the split) is
controlled by `seed` in the config through `src.utils.set_seed`. cuDNN is set to
deterministic mode. Runs on Apple MPS may still not be bit-exact.

## Limitations

- **Patient IDs are inferred from file names.** The dataset ships no official patient
  metadata. The split assumes the file-name convention identifies patients; this has not been
  verified against the original source. With `split.mode: image` (the interim report's
  setting), images of the same patient can fall into both train and validation, which makes
  validation scores optimistic. In the original train folder, 1,005 of 1,635 pneumonia
  patient IDs have more than one image.
- **Approximate 80/20 split.** The patient-level split (StratifiedGroupKFold, first of 5 folds)
  guarantees the class ratio and the 80/20 share only approximately (seed 42 with
  scikit-learn ≥ 1.8: 4,172 train / 1,044 val, 74.3% / 74.2% pneumonia). Older scikit-learn
  versions produce a different split for the same seed. With the image-level split, 469 of
  the 847 patient IDs in validation also appear in training.
- **Aspect-ratio distortion.** Images vary widely in size and aspect ratio and are resized
  directly to 227 × 227 without cropping or padding.
- **Channel-format artifact.** 283 training pneumonia images are stored as RGB; all other
  images are grayscale. All images are converted to grayscale first so that the channel
  format cannot act as a shortcut feature.
- **Train/test overlap is unverified.** Some patient IDs appear in both the original train and
  test folders. It is unknown whether these are the same patients or whether the numbering
  restarts per folder. The original test split is kept unchanged for comparability with
  published work.
- **Single source, pediatric only.** All images come from one center (Guangzhou Women and
  Children's Medical Center) and from patients aged 1–5, which limits how well results
  generalize to adults or other devices and sites.
- **Resolution mismatch.** torchvision's AlexNet was pretrained at 224 × 224; 227 × 227 is
  used to match the interim report (it works because of the adaptive pooling layer).

## License

[MIT](LICENSE)
