"""End-to-end tests of training, resume, and evaluation on the synthetic dataset (CPU)."""

import csv
import json
from pathlib import Path

import pytest
import torch

from src.checkpoint import load_checkpoint
from src.runner import HISTORY_FIELDS, evaluate_run, run_training
from src.utils import load_config
from tests.conftest import BASELINE_CONFIG

RUN_FILES = [
    "config.yaml",
    "run_info.json",
    "history.csv",
    "last.pt",
    "val_metrics.json",
    "val_predictions.csv",
    "curves.png",
    "confusion_val.png",
    "roc_val.png",
]


def _run_dir(cfg, name):
    return Path(cfg["output_dir"]) / name


def _assert_same_weights(path_a, path_b):
    a, b = load_checkpoint(path_a)["model"], load_checkpoint(path_b)["model"]
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory, data_root):
    """One finished 2-epoch run, shared by the tests that only read its outputs."""
    cfg = load_config(BASELINE_CONFIG)
    cfg["data"].update(root=str(data_root), num_workers=0, cache=False, image_size=64)
    cfg["split"]["file"] = None
    cfg["model"]["pretrained"] = False
    cfg["train"].update(batch_size=8, epochs=2)
    run_dir = tmp_path_factory.mktemp("outputs") / "run"
    summary = run_training(cfg, run_dir, device="cpu")
    return cfg, run_dir, summary


def test_training_writes_all_outputs(completed_run):
    _, run_dir, summary = completed_run
    assert summary["completed"] and summary["epoch"] == 2
    for name in RUN_FILES:
        assert (run_dir / name).is_file(), name

    with open(run_dir / "history.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    assert list(rows[0]) == HISTORY_FIELDS

    report = json.loads((run_dir / "val_metrics.json").read_text(encoding="utf-8"))
    assert report["epoch"] == 2  # the final-epoch model is the result
    assert report["metrics"]["error"] == pytest.approx(float(rows[-1]["val_error"]))
    assert 0.0 <= report["metrics"]["accuracy"] <= 1.0


def test_val_predictions_match_metrics(completed_run):
    cfg, run_dir, summary = completed_run
    with open(run_dir / "val_predictions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    correct = sum(int(r["label"]) == int(r["pred"]) for r in rows)
    assert correct / len(rows) == pytest.approx(summary["val_metrics"]["accuracy"])
    assert all(r["path"].startswith("train/") for r in rows)  # val comes from train/


def test_existing_run_is_not_overwritten(completed_run):
    cfg, run_dir, _ = completed_run
    with pytest.raises(FileExistsError):
        run_training(cfg, run_dir, device="cpu")


def test_resume_gives_the_same_model_as_an_uninterrupted_run(train_cfg):
    run_training(train_cfg, _run_dir(train_cfg, "straight"), device="cpu")

    interrupted = _run_dir(train_cfg, "interrupted")
    first = run_training(train_cfg, interrupted, device="cpu", stop_after=1)
    assert first == {"epoch": 1, "completed": False, "val_metrics": first["val_metrics"]}
    assert load_checkpoint(interrupted / "last.pt")["epoch"] == 1
    second = run_training(train_cfg, interrupted, resume=True, device="cpu")
    assert second["completed"] and second["epoch"] == 2

    _assert_same_weights(_run_dir(train_cfg, "straight") / "last.pt", interrupted / "last.pt")
    straight_history = load_checkpoint(_run_dir(train_cfg, "straight") / "last.pt")["history"]
    resumed_history = load_checkpoint(interrupted / "last.pt")["history"]
    for a, b in zip(straight_history, resumed_history, strict=True):
        assert {k: v for k, v in a.items() if k != "epoch_time_s"} == {
            k: v for k, v in b.items() if k != "epoch_time_s"
        }


def test_resume_without_checkpoint_raises(train_cfg):
    with pytest.raises(FileNotFoundError):
        run_training(train_cfg, _run_dir(train_cfg, "missing"), resume=True, device="cpu")


def test_evaluate_run_on_val_reproduces_the_training_metrics(completed_run):
    _, run_dir, summary = completed_run
    metrics = evaluate_run(run_dir, split="val", device="cpu")
    assert metrics == pytest.approx(summary["val_metrics"])
    assert (run_dir / "eval_val" / "metrics.json").is_file()


def test_evaluate_run_refuses_test_without_final(completed_run):
    _, run_dir, _ = completed_run
    with pytest.raises(ValueError, match="final evaluation"):
        evaluate_run(run_dir, split="test", device="cpu")
    assert not (run_dir / "eval_test").exists()


def test_evaluate_run_on_test_with_final(completed_run):
    # Synthetic data only: checks that the explicit final-evaluation path works.
    _, run_dir, _ = completed_run
    metrics = evaluate_run(run_dir, split="test", final=True, device="cpu")
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == 7
    assert (run_dir / "eval_test" / "test_predictions.csv").is_file()


def test_evaluate_run_refuses_an_unfinished_run(train_cfg):
    run_dir = _run_dir(train_cfg, "unfinished")
    run_training(train_cfg, run_dir, device="cpu", stop_after=1)
    with pytest.raises(RuntimeError, match="resume it"):
        evaluate_run(run_dir, device="cpu")
