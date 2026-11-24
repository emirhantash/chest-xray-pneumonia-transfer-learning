"""Binary classification metrics with PNEUMONIA (label 1) as the positive class."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _ratio(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 for an empty denominator (sklearn's zero_division=0)."""
    return float(numerator / denominator) if denominator else 0.0


def confusion_counts(y_true, y_pred) -> dict:
    """Return the confusion-matrix cells ``tn, fp, fn, tp`` as ints."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def binary_metrics(y_true, y_prob, threshold: float, y_score=None) -> dict:
    """Confusion matrix and summary metrics from labels and predicted P(pneumonia).

    A sample is predicted as pneumonia when ``y_prob >= threshold``. ``error`` is
    ``1 - accuracy`` (the HPO objective). The AUC is computed from ``y_score`` when it is
    given (the log-odds, which do not saturate like float probabilities near 1.0) and
    from ``y_prob`` otherwise, and so is ``pr_auc`` (average precision, PNEUMONIA
    positive). ``brier`` is the mean squared error of ``y_prob``. These are NaN when only
    one class is present or when a score is NaN (a diverged model); NaN probabilities count
    as NORMAL.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    score = y_prob if y_score is None else np.asarray(y_score, dtype=float)
    counts = confusion_counts(y_true, (y_prob >= threshold).astype(int))
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]

    accuracy = _ratio(tp + tn, len(y_true))
    recall = _ratio(tp, tp + fn)  # sensitivity
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    auc_defined = len(np.unique(y_true)) == 2 and bool(np.all(np.isfinite(score)))
    return {
        **counts,
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "balanced_accuracy": (recall + specificity) / 2,
        "auc": float(roc_auc_score(y_true, score)) if auc_defined else float("nan"),
        "pr_auc": float(average_precision_score(y_true, score)) if auc_defined else float("nan"),
        "brier": float(np.mean((y_prob - y_true) ** 2))
        if np.all(np.isfinite(y_prob))
        else float("nan"),
    }


def roc_points(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    """False-positive and true-positive rates of the ROC curve (scores: log-odds or
    probabilities; any monotone score gives the same curve, up to ties)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr
