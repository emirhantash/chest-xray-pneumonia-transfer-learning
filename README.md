# Pneumonia Detection from Pediatric Chest X-Rays with AlexNet Transfer Learning and Hyperparameter Optimization

[![CI](https://github.com/emirhantash/chest-xray-pneumonia-transfer-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/emirhantash/chest-xray-pneumonia-transfer-learning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

BİM 541 course project (Erciyes University). Binary classification (Normal vs Pneumonia) on
the Kaggle "Chest X-Ray Images (Pneumonia)" dataset with an ImageNet-pretrained AlexNet
(PyTorch), followed by hyperparameter optimization (Bayesian optimization vs. Random Search).

**Status:** Data pipeline, training/evaluation pipeline (resumable, with a Colab/Kaggle
runner notebook), hyperparameter-optimization pipeline, test suite, and CI are done. The HPO
studies are running; results and the final test evaluation come next.

## Setup

Requires Python ≥ 3.11 and scikit-learn ≥ 1.8. The train/val split is read from a file
committed to the repository (`splits/patient_seed42.csv`), so it is the same on every
machine; recomputing it requires scikit-learn ≥ 1.8 (see
[D9](docs/deviations_from_interim_report.md#d9-the-patient-level-split-depends-on-the-scikit-learn-version-decided-phase-15)).

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
  `NORMAL2-IM-<ID>-...`), so all images of a patient end up in the same subset. The
  resulting assignment (4,172 train / 1,044 val images) is saved in
  `splits/patient_seed42.csv` (`split.file`) and read from there; `scripts/make_split.py`
  regenerates it.
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

## Training

```bash
python scripts/train.py --config configs/baseline.yaml --data-root <path>
```

This trains for a fixed number of epochs (`train.epochs`) on the training subset and
evaluates on the validation subset after every epoch. The model after the **final epoch**
is the result of the run: there is no early stopping and no best-epoch selection (see
[D10](docs/deviations_from_interim_report.md#d10-training-loop-and-model-selection-decided-phase-2)).
The original test set is not touched.

Outputs go to `outputs/<run_name>/` (`--output-dir` and `--run-name` override the config):

| File | Content |
|---|---|
| `config.yaml`, `run_info.json` | effective config; library versions, device, split hash |
| `history.csv` | per-epoch train loss/accuracy and validation loss, accuracy, error, recall, specificity, precision, F1, balanced accuracy, AUC |
| `last.pt` | checkpoint after the latest epoch (≈435 MB; the final model once the run is complete) |
| `val_metrics.json`, `val_predictions.csv` | final-model validation metrics and per-image predictions |
| `curves.png`, `confusion_val.png`, `roc_val.png` | learning curves, confusion matrix, ROC curve |

**Resuming.** A checkpoint (model, optimizer, AMP scaler, history, RNG states) is written
atomically after every epoch. After an interruption, run the same command with `--resume`
to continue from the last completed epoch; the resumed run uses the config stored in the
checkpoint and gives the same weights as an uninterrupted run on the same device (tested
on CPU).

**Speed and memory.** Mixed precision (`train.amp`) is used on CUDA only; MPS and CPU train
in float32. With `data.cache: true` the resized grayscale images are kept in RAM
(≈270 MB); without worker processes (`data.num_workers: 0`) this is ~3× faster than with
workers on Windows. Training needs under 1 GB of GPU memory at batch size 8–64 (AMP).

**Colab / Kaggle.** [`notebooks/colab_kaggle_runner.ipynb`](notebooks/colab_kaggle_runner.ipynb)
clones the repository, installs the dependencies, locates the dataset, writes outputs to
Google Drive (Colab) or `/kaggle/working` (Kaggle), and resumes automatically when a
checkpoint exists.

## Evaluation

```bash
python scripts/evaluate.py --run outputs/baseline                     # validation set
python scripts/evaluate.py --run outputs/baseline --split test --final
```

Metrics are computed with PNEUMONIA as the positive class: confusion matrix, accuracy,
error (1 − accuracy), recall (sensitivity), specificity, precision, F1, balanced accuracy,
and ROC AUC. A sample is predicted as pneumonia when P(pneumonia) ≥ 0.5
(`eval.threshold`, fixed). The **test set is evaluated only with an explicit
`--split test --final`**, once, for the final models; it is never used to select models,
epochs, thresholds, or hyperparameters.

## Hyperparameter optimization

Bayesian optimization with a Gaussian process (Optuna `GPSampler`) is compared with Random
Search (`RandomSampler`) under the same conditions (`configs/hpo.yaml`, details in
[D13](docs/deviations_from_interim_report.md#d13-hpo-protocol-decided-phase-3)):

| | |
|---|---|
| Search space | lr [1e-4, 1e-2] log, batch size {8, 16, 32, 64}, epochs [5, 30], weight decay [1e-5, 1e-3] log, momentum [0.8, 0.99] |
| Objective | validation error (1 − accuracy) of the final-epoch model; a failed trial (diverged, or predicting one class for every image) gets the error of the majority-class predictor (0.2577) |
| Budget | 30 completed trials per study; sampler seeds 0, 1, 2 for each method |
| Fixed | training seed, split, preprocessing, class weights, AMP; no pruning |

```bash
python scripts/hpo.py --method gp --seed 0 --data-root <path>       # Bayesian optimization
python scripts/hpo.py --method random --seed 0 --data-root <path>   # Random Search
python scripts/compare_hpo.py                                       # summary table and plots
python scripts/train.py --config outputs/hpo/gp_seed0/best_config.yaml --data-root <path>
```

Each study lives in `outputs/hpo/<method>_seed<seed>/` (`study.db`, `trials.csv`,
`best_config.yaml`). Rerunning the same command resumes an interrupted study at the last
completed trial; the interrupted trial is repeated with the same parameters. With
`--backup-dir`, the SQLite file stays on local disk and is copied (e.g. to Google Drive)
after every trial. `compare_hpo.py` writes `summary.csv` (best, mean of the five best and
median error, trials that beat the baseline, failed trials, best error of the random
initial design and of the later trials), the best-so-far curves of both methods with the baseline as
reference, all trial errors, and each hyperparameter vs the error. With the same seed,
both methods share their first five (random) trials, so the comparison is paired. On the development GPU (RTX 3050 Ti, AMP, image cache) one epoch takes 9–18 s
depending on the batch size, i.e. roughly 2 hours per study.

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
configs/hpo.yaml           # search space and budget (interim report, Table 4)
splits/patient_seed42.csv  # the train/val assignment (file names only)
src/utils.py               # set_seed, seed_worker, load_config, get_device
src/data.py                # file listing, patient IDs, split (+ split file), transforms,
                           #   image cache, loaders, class weights
src/model.py               # build_alexnet
src/metrics.py             # confusion matrix, accuracy, recall, specificity, F1, AUC, ROC
src/engine.py              # optimizer, loss, train / evaluate loops, AMP
src/checkpoint.py          # atomic checkpoints, RNG state capture/restore
src/plots.py               # learning curves, confusion matrix, ROC figures
src/runner.py              # shared epoch loop, resumable training run, evaluation of a run
src/hpo.py                 # Optuna studies (GP vs Random), resume logic, analysis
scripts/check_data.py      # data/split/model sanity checks
scripts/make_split.py      # computes and saves the split file
scripts/train.py           # training entry point
scripts/evaluate.py        # evaluation entry point (test set only with --final)
scripts/hpo.py             # runs or resumes one HPO study
scripts/compare_hpo.py     # compares the studies (table and plots)
notebooks/colab_kaggle_runner.ipynb   # runs the pipeline on Colab / Kaggle GPUs
tests/                     # pytest suite on synthetic data (fixtures in conftest.py)
docs/deviations_from_interim_report.md   # what changed since the interim report and why
pyproject.toml             # project metadata, ruff and pytest settings
requirements.txt           # runtime dependencies (install PyTorch first)
requirements-dev.txt       # pytest, ruff
.github/workflows/ci.yml   # CI: lint + tests on CPU
```

## Reproducibility

All randomness (Python, NumPy, PyTorch, DataLoader shuffling and workers, the split) is
controlled by `seed` in the config through `src.utils.set_seed`. The shuffling generator
is re-seeded with `seed + epoch` at the start of every epoch, so a resumed run sees the
same batches as an uninterrupted one. cuDNN is set to deterministic mode, but some CUDA
kernels are not bit-exact between runs, and runs on Apple MPS may not be either. Library
versions, the device, and the git commit of every run are saved in `run_info.json` (and
with every HPO session).

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
