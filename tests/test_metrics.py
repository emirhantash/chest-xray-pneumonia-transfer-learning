"""Tests for src/metrics.py against hand-computed values."""

import math

import numpy as np
import pytest

from src.metrics import binary_metrics, confusion_counts, roc_points

# 10 samples: 6 pneumonia (1), 4 normal (0).
Y_TRUE = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
Y_PROB = [0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.7, 0.3, 0.2, 0.1]
# At threshold 0.5: predictions 1 1 1 1 0 0 | 1 0 0 0 -> tp 4, fn 2, fp 1, tn 3.


def test_confusion_counts():
    y_pred = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]
    assert confusion_counts(Y_TRUE, y_pred) == {"tn": 3, "fp": 1, "fn": 2, "tp": 4}


def test_binary_metrics_hand_computed():
    m = binary_metrics(Y_TRUE, Y_PROB, threshold=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 2, 4)
    assert m["accuracy"] == pytest.approx(7 / 10)
    assert m["error"] == pytest.approx(3 / 10)
    assert m["recall"] == pytest.approx(4 / 6)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["precision"] == pytest.approx(4 / 5)
    assert m["f1"] == pytest.approx(2 * (4 / 5) * (4 / 6) / (4 / 5 + 4 / 6))
    assert m["balanced_accuracy"] == pytest.approx((4 / 6 + 3 / 4) / 2)
    # AUC = fraction of (pneumonia, normal) pairs ranked correctly, ties count 1/2:
    # 0.9, 0.8 beat all 4; 0.7 beats 3 and ties 1; 0.6 and 0.4 beat 3; 0.2 beats 1, ties 1.
    assert m["auc"] == pytest.approx((4 + 4 + 3.5 + 3 + 3 + 1.5) / 24)


def test_threshold_is_inclusive():
    m = binary_metrics([1, 0], [0.5, 0.49], threshold=0.5)
    assert (m["tp"], m["tn"]) == (1, 1)


def test_perfect_predictions():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    for key in ("accuracy", "recall", "specificity", "precision", "f1", "auc"):
        assert m[key] == pytest.approx(1.0)
    assert m["error"] == pytest.approx(0.0)


def test_zero_divisions_return_zero():
    # Everything predicted as normal: no positive predictions -> precision 0/0.
    m = binary_metrics([0, 1, 1], [0.1, 0.2, 0.3], threshold=0.5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["specificity"] == 1.0


def test_auc_is_nan_for_a_single_class():
    m = binary_metrics([1, 1, 1], [0.2, 0.6, 0.9], threshold=0.5)
    assert math.isnan(m["auc"])
    assert m["specificity"] == 0.0  # no negatives


def test_roc_points_span_the_unit_square():
    fpr, tpr = roc_points(Y_TRUE, Y_PROB)
    assert fpr[0] == 0 and tpr[0] == 0
    assert fpr[-1] == 1 and tpr[-1] == 1
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
