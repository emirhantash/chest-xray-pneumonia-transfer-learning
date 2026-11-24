# Results

Small, final result tables, copied from `outputs/` by `scripts/export_results.py`.
Checkpoints, study databases and per-image predictions are not in the repository.
All numbers were measured on the local GPU (RTX 3050 Ti, torch 2.15, see
`requirements-lock.txt`).

| File | Content |
|---|---|
| `hpo/<method>_seed<n>_trials.csv` | Every completed trial of one study in order (`trial` = completed-trial index, 0-based): `val_error` after the failed-trial rule (D13), the `raw_value` stored by Optuna, `failed`, the five hyperparameters, training time |
| `hpo/summary.csv` | Per-study summary from `scripts/compare_hpo.py` (best, mean of 5 best, median, failed trials, initial design vs later trials, trials that beat the baseline) |
| `final/model_metrics.csv` | Test metrics of the baseline and the six retrained best models with 95 % patient-level bootstrap intervals, the paired accuracy difference to the baseline and the exact McNemar test |
| `final/method_summary.csv` | Mean ± standard deviation over the three seeds of each method |
| `final/patient_metrics.csv` | Test metrics at the patient level (a patient's prediction is the maximum over its images) |
| `final/results.md` | The final tables in Markdown |

The figures are in [`docs/figures/`](../docs/figures/). The protocol behind every number is
in [`docs/deviations_from_interim_report.md`](../docs/deviations_from_interim_report.md).
