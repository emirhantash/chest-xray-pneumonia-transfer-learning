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

## D6. Bayesian optimization method (decided, Phase 3)

- **Interim report:** MATLAB `bayesopt` (Gaussian-process-based Bayesian optimization).
- **Now:** Optuna's `GPSampler` (Gaussian-process Bayesian optimization; Optuna 5.1) versus
  Optuna's `RandomSampler` (Random Search). The search space is the interim report's
  Table 4; `batch_size` is categorical and `epochs` an integer, both supported by
  `GPSampler`.
- **Why GP and not TPE:** Optuna's default sampler (TPE) is also a Bayesian, model-based
  method, but it is not GP-based. `GPSampler` stays closer to the method proposed in the
  interim report. TPE could be added later as a third method under the same protocol.
- **Differences from `bayesopt` that remain:** the acquisition function and its
  optimizer, the kernel and its hyperparameter priors, and the number of random initial
  points (5 here, `bayesopt` default 4) are Optuna's, not MATLAB's.

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
- **Follow-up (decided, Phase 2):** The reference split is saved as
  `splits/patient_seed42.csv` (one row per image of the original train folder: subset,
  relative path, label, patient ID; no images) by `scripts/make_split.py`, and training
  reads it through `split.file` instead of recomputing it. The split therefore no longer
  depends on the installed scikit-learn version. Loading fails if the file does not list
  exactly the images in the train folder or if a label disagrees with its class folder.
  `scripts/check_data.py` also recomputes the split and reports whether the installed
  scikit-learn reproduces the file (it does with 1.9.1). Every checkpoint stores a hash of
  the train/val assignment, and resuming or evaluating a run with a different split is
  refused.

## D10. Training loop and model selection (decided, Phase 2)

- **Interim report:** Table 3 gives the optimizer settings and a *maximum* of 15 epochs; no
  learning-rate schedule, early stopping, or validation-based checkpoint selection is
  described. With MATLAB's `trainNetwork` defaults this means a constant learning rate and
  the network of the last iteration.
- **Now:** A fixed number of epochs (`train.epochs`) with a constant learning rate. The model
  after the final epoch is the result of a run; there is no early stopping and no "best
  epoch" checkpoint. The epoch with the lowest validation error is logged in
  `val_metrics.json` for information only.
- **Why:** The number of epochs is one of the hyperparameters searched in Phase 3. Selecting
  the best validation epoch would make that hyperparameter meaningless and would also bias
  the validation error (the HPO objective) downward.
- **Details that the interim report leaves open, now fixed:**
  - Validation error = 1 − validation accuracy, with a sample predicted as pneumonia when
    P(pneumonia) ≥ 0.5 (`eval.threshold`). The threshold is fixed and never tuned.
  - The training data is reshuffled every epoch (MATLAB's default is `Shuffle="once"`).
  - L2 weight decay is applied to all parameters including biases (MATLAB's default
    `BiasL2Factor = 0` exempts biases). The effect is expected to be negligible.
  - The logged loss is the class-weighted cross-entropy, averaged exactly over the epoch.
  - The training loss and accuracy are running values in training mode (dropout active),
    as is usual for training curves; validation metrics come from a separate evaluation
    pass in eval mode.

## D11. Mixed precision on CUDA (decided, Phase 2)

- **Interim report:** Not mentioned (MATLAB trains in single precision).
- **Now:** Automatic mixed precision (float16 autocast with gradient scaling) on CUDA GPUs
  (`train.amp: true`); MPS and CPU always train in float32.
- **Why:** Faster epochs on the notebook GPUs used for Phase 3, where many training runs
  are needed. On the development GPU (RTX 3050 Ti), AMP made a training step about 1.6×
  faster at batch size 64 and about the same at batch size 16 (synthetic-input benchmark,
  2026-11-21). The baseline and all HPO trials use the same setting, so the comparison
  between methods is not affected. Results on CUDA and on CPU/MPS are therefore not
  bit-identical.

## D12. In-memory image cache (decided, Phase 2)

- **Interim report:** Not applicable (MATLAB `imageDatastore` reads files on demand).
- **Now:** With `data.cache: true`, every train/val image is decoded, converted to
  grayscale, and resized once, then kept in RAM as uint8 (about 270 MB at 227 × 227).
- **Why:** Decoding and resizing the full-resolution JPEGs took 7.9 ms per image versus
  0.84 ms for the remaining transform on a cached image (measured on the development
  machine). This matters most for the many HPO runs in Phase 3. Without augmentation the
  cached path produces exactly the same tensors as reading the files; a test checks this.
  If augmentation is added later (D5), it must be applied after the cache.
- **DataLoader workers (Phase 3):** With the cache, worker processes only add
  inter-process copying of the batches. On the development machine (Windows), one epoch
  with 2 workers took ~35 s at batch size 16 and ~34 s at 64; without workers it took
  10.9 s and 8.7 s, with identical losses and metrics. `data.num_workers` is therefore 0
  (the Phase 2 baseline run used 2 workers; its results are unaffected).

## D13. HPO protocol (decided, Phase 3)

- **Interim report:** Bayesian optimization vs Random Search over the Table 4 search space,
  objective = 1 − validation accuracy; budget and repetitions not fixed.
- **Now:**
  - One study per (method, sampler seed); **30 completed trials** per study (MATLAB
    `bayesopt`'s default number of evaluations), sampler seeds **0, 1, 2** for both methods,
    so both methods get exactly the same budget and seeds.
  - Every trial trains with the base config's seed (42): same initialization and, for equal
    batch sizes, the same batch order. The objective therefore depends only on the
    hyperparameters (up to GPU non-determinism).
  - Objective: validation error of the trial's **final-epoch model** (D10); no early
    stopping and **no pruning**, so every trial is a complete training run and the two
    methods are not affected differently by a pruning rule.
  - A trial whose weights become NaN/inf stops early: they cannot recover, so the
    remaining epochs would not change the final model's (constant) predictions. Its error
    is recorded as is (NaN probabilities count as NORMAL).
  - Studies are resumable at trial granularity (SQLite storage per study). A trial cut off
    by a crash or a lost notebook session is marked failed and its parameters are queued
    again, so it is repeated rather than replaced by a new configuration. Samplers keep
    their random state in memory only; a resumed session derives its sampler seed from
    (seed, number of existing trials), so it never repeats earlier draws. An uninterrupted
    study is fully reproducible from its seed; a resumed one is not bit-identical to an
    uninterrupted one.
  - `GPSampler` is used with its default noise model (`deterministic_objective=False`),
    because GPU training is not bit-exact and many configurations are expected to reach
    nearly the same error.
- **Known limitation:** the validation set is small for this task (1,044 images, one image
  = 0.096 percentage points) and already saturated by the baseline (98.85 % accuracy).
  Differences between methods may be within noise; results are reported over seeds as
  best-so-far curves, the number of trials needed to beat the baseline, and the final test
  results of the selected models (Phase 4).

## D14. Final models of the HPO methods (decided, Phase 3)

- **Interim report:** Not specified.
- **Now:** Trial models are not saved (a checkpoint is ~435 MB). After a study, its best
  configuration is written to `best_config.yaml` and retrained once with
  `scripts/train.py`, which produces the full outputs (checkpoint, curves, predictions)
  used in Phase 4. The retrained model's validation error can differ slightly from the
  trial's because GPU training is not bit-exact; both values are reported.
