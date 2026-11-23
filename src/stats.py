"""Uncertainty and significance for the final test evaluation.

- :func:`cluster_bootstrap`: patient-level (cluster) bootstrap of the test metrics. Images
  of the same patient are not independent, so whole patients are resampled. All models
  share the same resamples, which makes paired differences between models meaningful.
- :func:`mcnemar_exact`: exact McNemar test on the paired correct/wrong decisions of two
  models on the same images.
"""

import math

import numpy as np

from src.metrics import binary_metrics

CI_METRICS = (
    "accuracy",
    "recall",
    "specificity",
    "precision",
    "f1",
    "balanced_accuracy",
    "auc",
)


def cluster_bootstrap(
    y_true,
    y_probs: dict[str, np.ndarray],
    groups,
    threshold: float,
    n_boot: int = 2000,
    seed: int = 0,
    metrics=CI_METRICS,
) -> dict[str, dict[str, np.ndarray]]:
    """Bootstrap distributions of ``metrics`` for every model in ``y_probs``.

    Each replicate draws as many groups (patients) as there are, with replacement, and
    takes all images of the drawn groups. Returns ``{model: {metric: array(n_boot)}}``;
    replicate ``i`` uses the same images for every model.
    """
    y_true = np.asarray(y_true, dtype=int)
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    rng = np.random.default_rng(seed)
    out = {name: {m: np.empty(n_boot) for m in metrics} for name in y_probs}
    for i in range(n_boot):
        drawn = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([members[g] for g in drawn])
        for name, prob in y_probs.items():
            values = binary_metrics(y_true[idx], np.asarray(prob)[idx], threshold)
            for m in metrics:
                out[name][m][i] = values[m]
    return out


def percentile_ci(samples, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided percentile interval of bootstrap samples (NaN replicates ignored)."""
    low, high = np.nanquantile(np.asarray(samples, dtype=float), [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact (binomial) two-sided McNemar test of two classifiers on the same samples.

    ``a_only`` = samples only model A gets right, ``b_only`` = only model B. Under the
    null hypothesis of equal error rates, ``a_only`` ~ Binomial(a_only + b_only, 0.5).
    """
    y_true = np.asarray(y_true, dtype=int)
    correct_a = np.asarray(pred_a, dtype=int) == y_true
    correct_b = np.asarray(pred_b, dtype=int) == y_true
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n = a_only + b_only
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1))
        p_value = min(1.0, 2 * tail / 2**n)
    return {"a_only": a_only, "b_only": b_only, "p_value": float(p_value)}
