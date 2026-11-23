"""Tests for src/stats.py: patient-level bootstrap and the exact McNemar test."""

import numpy as np
import pytest
from scipy.stats import binomtest

from src.stats import cluster_bootstrap, mcnemar_exact, percentile_ci

Y = np.array([0, 0, 1, 1, 1, 1])
GROUPS = np.array(["a", "a", "b", "b", "c", "d"])  # patient "a" and "b" have two images


def test_bootstrap_shapes_and_seed():
    probs = {"m1": np.array([0.1, 0.6, 0.9, 0.4, 0.8, 0.7]), "m2": np.full(6, 0.9)}
    first = cluster_bootstrap(Y, probs, GROUPS, 0.5, n_boot=50, seed=3)
    again = cluster_bootstrap(Y, probs, GROUPS, 0.5, n_boot=50, seed=3)
    assert set(first) == {"m1", "m2"}
    assert first["m1"]["accuracy"].shape == (50,)
    np.testing.assert_array_equal(first["m1"]["accuracy"], again["m1"]["accuracy"])


def test_perfect_model_has_a_degenerate_interval():
    probs = {"perfect": Y.astype(float)}
    samples = cluster_bootstrap(Y, probs, GROUPS, 0.5, n_boot=100, metrics=("accuracy",))
    assert percentile_ci(samples["perfect"]["accuracy"]) == (1.0, 1.0)


def test_identical_models_get_identical_replicates():
    prob = np.array([0.2, 0.7, 0.9, 0.3, 0.6, 0.4])
    samples = cluster_bootstrap(Y, {"a": prob, "b": prob.copy()}, GROUPS, 0.5, n_boot=100)
    for metric in samples["a"]:
        np.testing.assert_array_equal(samples["a"][metric], samples["b"][metric])


def test_whole_patients_are_resampled():
    # The model is wrong on exactly both images of patient "a" and right elsewhere. With
    # patient-level resampling each replicate contains 0, 2, 4, ... wrong images, so the
    # number of errors is always even.
    prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    samples = cluster_bootstrap(Y, {"m": prob}, GROUPS, 0.5, n_boot=200, metrics=("tn", "fp"))
    assert np.all(samples["m"]["tn"] == 0)
    assert np.all(samples["m"]["fp"] % 2 == 0)


def test_percentile_ci_ignores_nan():
    assert percentile_ci([np.nan, 1.0, 2.0, 3.0], alpha=0.0) == (1.0, 3.0)


@pytest.mark.parametrize("a_only,b_only", [(0, 0), (10, 0), (3, 7), (12, 5), (40, 25)])
def test_mcnemar_matches_the_binomial_test(a_only, b_only):
    both = 5
    y = np.ones(a_only + b_only + both, dtype=int)
    pred_a = np.r_[np.ones(a_only), np.zeros(b_only), np.ones(both)]
    pred_b = np.r_[np.zeros(a_only), np.ones(b_only), np.ones(both)]
    result = mcnemar_exact(y, pred_a, pred_b)
    assert (result["a_only"], result["b_only"]) == (a_only, b_only)
    expected = 1.0 if a_only + b_only == 0 else binomtest(a_only, a_only + b_only).pvalue
    assert result["p_value"] == pytest.approx(expected)
    assert mcnemar_exact(y, pred_b, pred_a)["p_value"] == pytest.approx(result["p_value"])
