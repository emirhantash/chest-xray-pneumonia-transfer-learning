"""Final test-set report: per-model metrics with patient-level bootstrap intervals,
paired comparisons with the baseline, and per-method summaries over seeds.

Works only on prediction files written by ``evaluate_run(split="test", final=True)``.
"""

import csv
import re

import numpy as np

from src.data import patient_id
from src.metrics import binary_metrics
from src.stats import CI_METRICS, cluster_bootstrap, mcnemar_exact, percentile_ci

REFERENCE = "baseline"
_BEST_RUN = re.compile(r"^best_(gp|random)_seed(\d+)$")
SUMMARY_METRICS = ("accuracy", "recall", "specificity", "precision", "f1", "auc")


def read_predictions(path) -> dict:
    """Predictions CSV of an evaluated subset (``path, label, prob_pneumonia, pred``)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "path": [r["path"] for r in rows],
        "label": np.array([int(r["label"]) for r in rows]),
        "prob": np.array([float(r["prob_pneumonia"]) for r in rows]),
        "pred": np.array([int(r["pred"]) for r in rows]),
    }


def run_method(run_name: str) -> tuple[str, int | None]:
    """(method, seed) of a run: ``baseline`` or a retrained ``best_<method>_seed<n>``."""
    match = _BEST_RUN.match(run_name)
    if match:
        return match.group(1), int(match.group(2))
    return run_name, None


def model_rows(
    preds: dict[str, dict],
    val_errors: dict[str, float | None],
    threshold: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> list[dict]:
    """One row per model: test metrics, 95 % patient-level bootstrap intervals, and the
    paired comparison with the baseline (McNemar test, accuracy difference interval)."""
    names = list(preds)
    first = preds[names[0]]
    for name in names[1:]:
        if preds[name]["path"] != first["path"] or not np.array_equal(
            preds[name]["label"], first["label"]
        ):
            raise ValueError(f"{name} was evaluated on different images than {names[0]}.")
    y_true = first["label"]
    groups = [patient_id(p) for p in first["path"]]
    boot = cluster_bootstrap(
        y_true, {n: preds[n]["prob"] for n in names}, groups, threshold, n_boot, seed
    )

    rows = []
    for name in names:
        method, run_seed = run_method(name)
        metrics = binary_metrics(y_true, preds[name]["prob"], threshold)
        row = {"model": name, "method": method, "seed": run_seed, "val_error": val_errors[name]}
        row.update({k: metrics[k] for k in ("tn", "fp", "fn", "tp")})
        for m in CI_METRICS:
            low, high = percentile_ci(boot[name][m])
            row.update({m: metrics[m], f"{m}_ci_low": low, f"{m}_ci_high": high})
        if REFERENCE in preds and name != REFERENCE:
            test = mcnemar_exact(y_true, preds[name]["pred"], preds[REFERENCE]["pred"])
            diff = boot[name]["accuracy"] - boot[REFERENCE]["accuracy"]
            low, high = percentile_ci(diff)
            ref_accuracy = binary_metrics(y_true, preds[REFERENCE]["prob"], threshold)["accuracy"]
            row.update(
                {
                    "accuracy_diff_vs_baseline": metrics["accuracy"] - ref_accuracy,
                    "accuracy_diff_ci_low": low,
                    "accuracy_diff_ci_high": high,
                    "only_this_correct": test["a_only"],
                    "only_baseline_correct": test["b_only"],
                    "mcnemar_p": test["p_value"],
                }
            )
        rows.append(row)
    return rows


def method_rows(rows: list[dict]) -> list[dict]:
    """Mean and standard deviation (ddof=1) of the test metrics per method over seeds."""
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        group = [r for r in rows if r["method"] == method]
        summary = {"method": method, "n_models": len(group)}
        for m in ("val_error", *SUMMARY_METRICS):
            values = np.array([r[m] for r in group if r[m] is not None], dtype=float)
            summary[f"{m}_mean"] = float(values.mean()) if values.size else None
            summary[f"{m}_std"] = float(values.std(ddof=1)) if values.size > 1 else None
        out.append(summary)
    return out


def _pct(value) -> str:
    return "–" if value is None else f"{100 * value:.1f}"


def markdown_tables(rows: list[dict], methods: list[dict]) -> str:
    """Results as Markdown (per model, then per method), percentages with one decimal."""
    lines = [
        "| Model | Val error (%) | Test accuracy (%) [95% CI] | Recall | Specificity "
        "| Precision | F1 | AUC | McNemar p vs baseline |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        acc = f"{_pct(r['accuracy'])} [{_pct(r['accuracy_ci_low'])}, {_pct(r['accuracy_ci_high'])}]"
        p_value = r.get("mcnemar_p")
        lines.append(
            f"| {r['model']} | {_pct(r['val_error'])} | {acc} | {_pct(r['recall'])} "
            f"| {_pct(r['specificity'])} | {_pct(r['precision'])} | {_pct(r['f1'])} "
            f"| {r['auc']:.4f} | {'–' if p_value is None else f'{p_value:.3g}'} |"
        )
    lines += [
        "",
        "| Method | Models | Test accuracy (%) | Recall | Specificity | F1 | AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in methods:

        def cell(metric, scale=100.0, digits=1, m=m):
            mean, std = m[f"{metric}_mean"], m[f"{metric}_std"]
            text = f"{scale * mean:.{digits}f}"
            return text if std is None else f"{text} ± {scale * std:.{digits}f}"

        lines.append(
            f"| {m['method']} | {m['n_models']} | {cell('accuracy')} | {cell('recall')} "
            f"| {cell('specificity')} | {cell('f1')} | {cell('auc', 1.0, 4)} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
