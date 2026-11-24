"""Tests for the final test-set report (src/report.py, scripts/final_eval.py) on synthetic data."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from src.plots import plot_roc_overlay
from src.report import (
    markdown_tables,
    method_rows,
    model_rows,
    patient_level,
    patient_rows,
    read_predictions,
    run_method,
)
from src.runner import run_training

PATHS = [f"test/NORMAL/IM-000{i}-0001.jpeg" for i in range(4)] + [
    f"test/PNEUMONIA/person{i}_virus_{i}.jpeg" for i in range(6)
]
LABELS = np.array([0] * 4 + [1] * 6)


def _pred(prob):
    prob = np.asarray(prob, dtype=float)
    return {"path": list(PATHS), "label": LABELS, "prob": prob, "pred": (prob >= 0.5).astype(int)}


def _preds():
    return {
        "baseline": _pred([0.1, 0.2, 0.7, 0.8, 0.9, 0.9, 0.9, 0.6, 0.4, 0.9]),  # 3 wrong
        "best_gp_seed0": _pred([0.1, 0.2, 0.3, 0.8, 0.9, 0.9, 0.9, 0.6, 0.9, 0.9]),  # 1 wrong
        "best_gp_seed1": _pred([0.1, 0.2, 0.3, 0.4, 0.9, 0.9, 0.9, 0.6, 0.9, 0.9]),  # perfect
        "best_random_seed0": _pred([0.1, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.6, 0.4, 0.9]),
    }


def test_run_method():
    assert run_method("baseline") == ("baseline", None)
    assert run_method("best_gp_seed2") == ("gp", 2)
    assert run_method("best_random_seed0") == ("random", 0)


def test_read_predictions(tmp_path):
    path = tmp_path / "test_predictions.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "prob_pneumonia", "pred"])
        writer.writerows([["test/NORMAL/a.jpeg", 0, "0.250000", 0], ["test/P/b.jpeg", 1, "0.9", 1]])
    pred = read_predictions(path)
    assert pred["path"] == ["test/NORMAL/a.jpeg", "test/P/b.jpeg"]
    np.testing.assert_array_equal(pred["label"], [0, 1])
    np.testing.assert_allclose(pred["prob"], [0.25, 0.9])
    np.testing.assert_allclose(pred["score"], pred["prob"])  # no log-odds column

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "prob_pneumonia", "logit_pneumonia", "pred"])
        writer.writerows([["a", 0, "1.000000", "17.5", 1], ["b", 1, "1.000000", "21.25", 1]])
    np.testing.assert_allclose(read_predictions(path)["score"], [17.5, 21.25])


def test_model_rows_auc_from_log_odds():
    preds = _preds()
    for p in preds.values():  # saturated probabilities: every positive call ties at 1.0
        p["score"] = np.log(p["prob"] / (1 - p["prob"]))
        p["prob"] = np.where(p["prob"] >= 0.5, 1.0, 0.0)
    rows = model_rows(preds, dict.fromkeys(preds, None), 0.5, n_boot=20)
    # best_gp_seed0: one normal (log-odds of 0.8) outranks one pneumonia (0.6) -> 23/24;
    # from the tied probabilities it would be 21/24.
    assert rows[1]["auc"] == pytest.approx(23 / 24)
    assert rows[1]["accuracy"] == pytest.approx(0.9)


def test_model_rows():
    val_errors = {"baseline": 0.0115, "best_gp_seed0": 0.0077, "best_gp_seed1": 0.0077}
    val_errors["best_random_seed0"] = 0.0086
    rows = model_rows(_preds(), val_errors, 0.5, n_boot=200)
    by_name = {r["model"]: r for r in rows}
    base, gp0 = by_name["baseline"], by_name["best_gp_seed0"]
    assert base["accuracy"] == pytest.approx(0.7) and gp0["accuracy"] == pytest.approx(0.9)
    assert "mcnemar_p" not in base  # the reference is not compared with itself
    assert (gp0["only_this_correct"], gp0["only_baseline_correct"]) == (2, 0)
    assert gp0["accuracy_diff_vs_baseline"] == pytest.approx(0.2)
    assert gp0["accuracy_ci_low"] <= gp0["accuracy"] <= gp0["accuracy_ci_high"]
    assert gp0["accuracy_diff_ci_low"] >= 0.0  # gp0 is never worse on any resample
    assert (gp0["method"], gp0["seed"], gp0["val_error"]) == ("gp", 0, 0.0077)
    assert by_name["best_gp_seed1"]["accuracy_ci_low"] == 1.0


def test_patient_level_takes_the_maximum_per_patient():
    pred = {
        "path": ["IM-0001-0001.jpeg", "IM-0001-0002.jpeg", "person5_virus_9.jpeg"],
        "label": np.array([0, 0, 1]),
        "prob": np.array([0.2, 0.7, 0.9]),
        "score": np.array([-1.4, 0.8, 2.2]),
    }
    p = patient_level(pred)
    assert p["path"] == ["IM-0001", "person5"]
    np.testing.assert_array_equal(p["label"], [0, 1])
    np.testing.assert_allclose(p["prob"], [0.7, 0.9])
    np.testing.assert_allclose(p["score"], [0.8, 2.2])
    rows = patient_rows({"m": pred}, 0.5)
    assert rows[0]["n_patients"] == 2 and (rows[0]["fp"], rows[0]["tp"]) == (1, 1)

    pred["label"] = np.array([0, 1, 1])  # one patient with both classes
    with pytest.raises(ValueError, match="both classes"):
        patient_level(pred)


def test_model_rows_refuses_different_images():
    preds = _preds()
    preds["best_gp_seed0"]["path"] = list(reversed(PATHS))
    with pytest.raises(ValueError, match="different images"):
        model_rows(preds, dict.fromkeys(preds, None), 0.5, n_boot=10)


def test_method_rows_and_markdown():
    preds = _preds()
    rows = model_rows(preds, dict.fromkeys(preds, 0.01), 0.5, n_boot=50)
    methods = {m["method"]: m for m in method_rows(rows)}
    assert methods["gp"]["n_models"] == 2
    assert methods["gp"]["accuracy_mean"] == pytest.approx(0.95)
    assert methods["gp"]["accuracy_std"] == pytest.approx(np.std([0.9, 1.0], ddof=1))
    assert methods["baseline"]["accuracy_std"] is None  # a single model
    text = markdown_tables(rows, list(methods.values()))
    assert "| best_gp_seed0 | 1.0 | 90.0 [" in text
    assert "| gp | 2 | 95.0 ± 7.1 |" in text


def test_plot_roc_overlay(tmp_path):
    curves = [{"label": "a", "fpr": [0, 0.1, 1], "tpr": [0, 0.9, 1]}]
    curves.append({"label": "b", "fpr": [0, 1], "tpr": [0, 1], "color": "C1", "linestyle": "--"})
    curves[0].update(point=(0.1, 0.9), point_label="operating point (threshold 0.5)")
    plot_roc_overlay(curves, tmp_path / "roc.png", "ROC")
    assert (tmp_path / "roc.png").stat().st_size > 0


def test_final_eval_script_end_to_end(train_cfg, monkeypatch, tmp_path):
    # Synthetic data only: two finished runs, then the script evaluates both on the
    # (synthetic) test set and writes the report.
    output_dir = Path(train_cfg["output_dir"])
    for name in ("baseline", "best_gp_seed0"):
        run_training(train_cfg, output_dir / name, device="cpu")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import final_eval

    out = tmp_path / "final"
    argv = ["final_eval.py", "--final", "--output-dir", str(output_dir), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv + ["--n-boot", "20", "--device", "cpu"])
    final_eval.main()
    names = ("model_metrics.csv", "method_summary.csv", "patient_metrics.csv", "results.md")
    for name in (*names, "roc_test.png"):
        assert (out / name).is_file(), name
    assert (output_dir / "best_gp_seed0" / "eval_test" / "test_predictions.csv").is_file()
    with open(out / "model_metrics.csv", newline="", encoding="utf-8") as f:
        assert [r["model"] for r in csv.DictReader(f)] == ["baseline", "best_gp_seed0"]

    # Without --final the script refuses to touch the test set.
    monkeypatch.setattr(sys, "argv", ["final_eval.py", "--output-dir", str(output_dir)])
    with pytest.raises(SystemExit):
        final_eval.main()
