# Deviations from the interim report

This file records every place where the implementation differs from the interim report
(`BIM541_AraRapor_EmirhanTas_BatuhanCakmak.pdf`) and why. It is the source for the
"changes since the interim report" discussion in the final report.

Status legend: **decided**, i.e. implemented; **open**, i.e. to be decided in a later phase.

---

## D1. Framework: MATLAB → PyTorch (decided, Phase 1)

- **Interim report:** MATLAB Deep Learning Toolbox (`fullyConnectedLayer(2)`, softmax and
  classification layers), Bayesian optimization with MATLAB `bayesopt`.
- **Now:** PyTorch + torchvision AlexNet; HPO will use Optuna (Phase 3).
- **Why:** Open-source, reproducible pipeline that runs on Colab/Kaggle GPUs and on any local
  machine (CUDA / MPS / CPU); Optuna offers both Bayesian (model-based) samplers and Random
  Search under the same API, so the two methods can be compared under identical conditions.
- **Consequence for the report:** The softmax and classification layers are not separate layers in
  PyTorch: the final `Linear(4096, 2)` outputs logits and `CrossEntropyLoss` applies
  log-softmax internally. The architecture is otherwise equivalent.

## D2. Train/validation split: image-level → patient-level (decided, Phase 1)

- **Interim report (§3.3, §3.4):** Stratified 80/20 split of the original train set at the
  *image* level, stating that patient IDs are not available, and noting that validation
  results may therefore be slightly optimistic.
- **Finding:** The file names do encode a patient identifier:
  - Pneumonia: `person<ID>_<bacteria|virus>_<imageNo>.jpeg`
  - Normal: `IM-<ID>-<imageNo>.jpeg` and `NORMAL2-IM-<ID>-<imageNo>.jpeg`

  Counted on the original train folder (2026-11-21):

  | Class | Images | Unique patient IDs | IDs with > 1 image | Max images per ID |
  |---|---|---|---|---|
  | NORMAL | 1,341 | 1,211 | 70 | 4 |
  | PNEUMONIA | 3,875 | 1,635 | 1,005 | 30 |

  An image-level random split would therefore place images of the same pneumonia patient
  in both the training and validation subsets for a large share of patients.
- **Now:** Patient-level stratified split with `sklearn.model_selection.StratifiedGroupKFold`
  (5 folds, first fold = validation, i.e. ≈20%), fixed seed. The image-level split is still
  available via `split.mode: image` in the config.
- **Why:** The validation error is the HPO objective. Patient leakage between train and val
  would inflate validation performance and could bias the HPO toward configurations that
  memorize patient-specific features (e.g., more epochs), which would not transfer to the
  test set.
- **Assumptions / limitations:**
  - The patient ID is inferred from the file-name convention; it has not been confirmed
    against official metadata. `IM-xxxx` and `NORMAL2-IM-xxxx` are treated as different
    ID namespaces.
  - The split is no longer exactly 80/20 by image count and class ratios are only
    approximately preserved (see resulting numbers in D3).

## D3. Split sizes and class weights (decided, Phase 1)

- **Interim report:** ≈4,173 train / 1,043 validation; class weights ≈1.94 (Normal) / 0.67
  (Pneumonia) computed from ≈1,073 Normal and 3,100 Pneumonia training images.
- **Now (patient-level, seed 42, scikit-learn ≥ 1.8):** computed with
  `src/data.py::build_splits` and `scripts/check_data.py` on the full dataset (2026-11-21,
  scikit-learn 1.9.1):

  | Subset | Normal | Pneumonia | Total | Share of orig. train | % Pneumonia | Patient IDs |
  |---|---|---|---|---|---|---|
  | train | 1,072 | 3,100 | 4,172 | 80.0% | 74.3% | 2,277 |
  | val | 269 | 775 | 1,044 | 20.0% | 74.2% | 569 |
  | test (original) | 234 | 390 | 624 | – | 62.5% | 427 |

  Class weights: Normal **1.9459**, Pneumonia **0.6729**. Shared patient IDs between train and
  val: **0**. These numbers depend on the scikit-learn version (see D9); an earlier draft of
  this file listed the scikit-learn ≤ 1.7 result (4,133 / 1,083, weights 1.9367 / 0.6740).
  With this seed the subset sizes (4,172 / 1,044) happen to equal those of the image-level
  split below, but the subsets themselves differ: the patient-level split has no shared
  patients.

- **For comparison (image-level, same seed):** 4,172 train (1,073 / 3,099) / 1,044 val
  (268 / 776). Weights 1.9441 / 0.6731. **469 of the 847 patient IDs in val also appear in
  train**, which quantifies the leakage that motivated D2. (sklearn rounds the test size up:
  ceil(0.2 × 5,216) = 1,044, so the interim report's 4,173 / 1,043 was off by one.)

## D4. Input size 227 × 227 with torchvision AlexNet (decided, Phase 1)

- **Interim report:** 227 × 227 × 3, the MATLAB AlexNet input size.
- **Now:** 227 × 227 kept (configurable via `data.image_size`).
- **Note:** torchvision's AlexNet weights were trained at 224 × 224. The network accepts
  227 × 227 because of its `AdaptiveAvgPool2d((6, 6))` layer, so this works, but it is a small
  mismatch from the pretraining resolution. 224 is a candidate alternative if needed.

## D5. Data augmentation (open)

- **Interim report:** Augmentation (random horizontal flip, small rotation) "if time permits".
- **Now:** Not implemented in Phase 1; train and eval transforms are identical.
- **Note for later:** Horizontal flips on chest X-rays swap the anatomical sides (heart
  position, side markers); small rotations/affine transforms are the safer choice.

## D6. Bayesian optimization method (open, Phase 3)

- **Interim report:** MATLAB `bayesopt` (Gaussian-process-based Bayesian optimization).
- **Plan:** Optuna. Its default sampler (TPE) is a Bayesian, model-based method, but it is
  not GP-based. If the report should match GP-based BO, use `optuna.samplers.GPSampler`
  (Optuna ≥ 3.6). To be decided in Phase 3 and stated explicitly in the report.

## D7. Data findings not mentioned in the interim report (decided, Phase 1)

- **RGB files only in train/PNEUMONIA:** 283 of the 3,875 train pneumonia images are stored
  as 3-channel RGB; every other image (all Normal and all test images) is single-channel.
  If the channel format reached the model, it could become a shortcut feature that does not
  exist in the test set. The pipeline converts every image to grayscale (`convert("L")`)
  before replicating it to 3 channels, which removes this difference.
- **Varying size and aspect ratio:** widths range from 384 to 2,916 px and heights from 127
  to 2,713 px (median 1,282 × 886). Resizing directly to 227 × 227, as in the interim report,
  distorts the aspect ratio. This is kept for consistency with the interim report and noted
  as a limitation.

## D8. Patient IDs shared between original train and test folders (open, unverified)

- `scripts/check_data.py` reports how many parsed patient IDs appear in both the original
  train and test folders (170 on 2026-11-21). It is **not verified** whether these are the
  same patients or whether the ID numbering simply restarts per folder. The original
  test set is used unchanged for comparability with the literature; this should be
  mentioned as a possible limitation rather than claimed as leakage.

## D9. The patient-level split depends on the scikit-learn version (decided, Phase 1.5)

- **Finding (2026-11-21):** With the same data, config, and seed (42), `StratifiedGroupKFold`
  (`shuffle=True`) returns a different first fold depending on the scikit-learn version.
  Measured with a stand-alone copy of the split logic:

  | scikit-learn | Train (Normal / Pneumonia) | Val | Class weights (N / P) |
  |---|---|---|---|
  | 1.5.2, 1.6.1, 1.7.2 | 4,133 (1,067 / 3,066) | 1,083 | 1.9367 / 0.6740 |
  | 1.8.0, 1.9.1 | 4,172 (1,072 / 3,100) | 1,044 | 1.9459 / 0.6729 |

  Both versions give 2,277 / 569 patients and 0 shared patients; only the assignment of
  patients to the folds differs. The image-level split (`train_test_split`) is the same
  in all tested versions.
- **Decision:** Use scikit-learn ≥ 1.8 (`requirements.txt`) and the split it produces as
  the reference split (numbers in D3). Python ≥ 3.11 follows from this requirement.
- **Why:** The validation set defines the HPO objective, so every machine (local, Colab,
  Kaggle) must produce exactly the same split. Preinstalled scikit-learn versions on
  notebook platforms can differ, so the minimum version is enforced instead of relying on
  whatever is installed. The newer version was chosen over pinning `< 1.8` because no
  training run depended on the old split yet, it keeps the project on a maintained release,
  and its split is closer to 80/20.
- **Open point:** A future scikit-learn release could change the split again. Saving the
  split as a file list (file names only, no images) and loading it instead of recomputing
  it would remove this dependency; to be decided in Phase 2.
