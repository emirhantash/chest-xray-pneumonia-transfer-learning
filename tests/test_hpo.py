"""Tests for src/hpo.py: search space, samplers, resumable studies, analysis (CPU)."""

import shutil

import numpy as np
import optuna
import pytest
import torch
import yaml
from optuna.trial import TrialState

from src.data import build_splits
from src.hpo import (
    _check_study_settings,
    apply_params,
    best_so_far,
    constant_predictor_error,
    has_diverged,
    load_study_results,
    make_sampler,
    mark_interrupted_trials,
    predicts_single_class,
    requeue_failed_trials,
    run_study,
    session_seed,
    suggest_params,
    summarize,
)
from src.plots import plot_best_so_far, plot_params_vs_error, plot_trial_errors
from src.utils import load_config
from tests.conftest import BASELINE_CONFIG, REPO_ROOT

HPO_CONFIG = REPO_ROOT / "configs" / "hpo.yaml"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _dummy_objective(space):
    def objective(trial):
        p = suggest_params(trial, space)
        return (np.log10(p["lr"]) + 2.5) ** 2 + abs(p["batch_size"] - 32) / 64

    return objective


def _sampled_params(method, seed, n=8):
    space = load_config(HPO_CONFIG)["search_space"]
    study = optuna.create_study(sampler=make_sampler(method, seed, n_startup_trials=3))
    study.optimize(_dummy_objective(space), n_trials=n)
    return [t.params for t in study.trials]


# --- search space and samplers -----------------------------------------------------------


def test_hpo_config_matches_protocol():
    # Interim report, Table 4 (guards against accidental edits).
    cfg = load_config(HPO_CONFIG)
    assert cfg["base_config"] == "configs/baseline.yaml"
    assert (cfg["n_trials"], cfg["n_startup_trials"]) == (30, 5)
    space = cfg["search_space"]
    assert space["lr"] == {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}
    assert space["batch_size"] == {"type": "categorical", "choices": [8, 16, 32, 64]}
    assert space["epochs"] == {"type": "int", "low": 5, "high": 30}
    assert space["weight_decay"] == {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}
    assert space["momentum"] == {"type": "float", "low": 0.8, "high": 0.99}


def test_sampled_params_respect_the_search_space():
    space = load_config(HPO_CONFIG)["search_space"]
    for params in _sampled_params("random", seed=0, n=40):
        assert 1e-4 <= params["lr"] <= 1e-2
        assert params["batch_size"] in (8, 16, 32, 64)
        assert isinstance(params["epochs"], int) and 5 <= params["epochs"] <= 30
        assert 1e-5 <= params["weight_decay"] <= 1e-3
        assert 0.8 <= params["momentum"] <= 0.99
        assert set(params) == set(space)


@pytest.mark.parametrize("method", ["gp", "random"])
def test_same_seed_gives_the_same_configurations(method):
    assert _sampled_params(method, seed=1) == _sampled_params(method, seed=1)
    assert _sampled_params(method, seed=1) != _sampled_params(method, seed=2)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        make_sampler("tpe", 0, 5)


def test_session_seed_is_deterministic_and_progress_dependent():
    assert session_seed(0, 0) == session_seed(0, 0)
    assert len({session_seed(0, n) for n in range(10)}) == 10
    assert session_seed(0, 3) != session_seed(1, 3)


def test_apply_params_copies_the_base_config():
    base = load_config(BASELINE_CONFIG)
    cfg = apply_params(base, {"lr": 0.005, "batch_size": 32})
    assert (cfg["train"]["lr"], cfg["train"]["batch_size"]) == (0.005, 32)
    assert (base["train"]["lr"], base["train"]["batch_size"]) == (0.001, 16)
    with pytest.raises(KeyError, match="dropout"):
        apply_params(base, {"dropout": 0.5})


# --- interrupted and failed trials -------------------------------------------------------


def test_interrupted_trial_is_failed_and_requeued_once(tmp_path):
    space = load_config(HPO_CONFIG)["search_space"]
    storage = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    study = optuna.create_study(
        storage=storage, study_name="s", sampler=make_sampler("random", 0, 5)
    )
    trial = study.ask()
    params = suggest_params(trial, space)  # the "process" dies here: trial stays RUNNING

    resumed = optuna.load_study(study_name="s", storage=storage)
    assert mark_interrupted_trials(resumed) == [0]
    assert resumed.trials[0].state == TrialState.FAIL
    assert requeue_failed_trials(resumed) == [0]
    assert requeue_failed_trials(resumed) == []  # not queued twice

    retry = resumed.ask()
    assert suggest_params(retry, space) == params
    assert retry.user_attrs["retry_of"] == 0


def test_resuming_with_different_settings_raises(tmp_path):
    study = optuna.create_study()
    study.set_user_attr("method", "random")
    study.set_user_attr("seed", 0)
    _check_study_settings(study, {"method": "random", "seed": 0})
    with pytest.raises(ValueError, match="method"):
        _check_study_settings(study, {"method": "gp", "seed": 0})


# --- end-to-end study on the synthetic data ----------------------------------------------


def _mini_configs(data_root, tmp_path):
    """Tiny search space and a train-only copy of the data (HPO must not need test/)."""
    root = tmp_path / "data"
    shutil.copytree(data_root / "train", root / "train")
    base = load_config(BASELINE_CONFIG)
    base["data"].update(root=str(root), num_workers=0, cache=True, image_size=64)
    base["split"]["file"] = None
    base["model"]["pretrained"] = False
    hpo = load_config(HPO_CONFIG)
    hpo["n_trials"] = 2
    hpo["n_startup_trials"] = 1
    hpo["search_space"] = {
        "lr": {"type": "float", "low": 1e-3, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [4, 8]},
        "epochs": {"type": "int", "low": 1, "high": 2},
    }
    return hpo, base


@pytest.fixture(scope="module")
def mini_study(tmp_path_factory, data_root):
    tmp_path = tmp_path_factory.mktemp("hpo")
    hpo, base = _mini_configs(data_root, tmp_path)
    study_dir, backup = tmp_path / "outputs" / "random_seed0", tmp_path / "drive" / "random_seed0"
    first = run_study(
        hpo, base, "random", 0, study_dir, n_trials=1, device="cpu", backup_dir=backup
    )
    assert len(first.trials) == 1
    study = run_study(hpo, base, "random", 0, study_dir, device="cpu", backup_dir=backup)
    return hpo, base, study, study_dir, backup


def test_study_completes_after_resume(mini_study):
    _, _, study, study_dir, _ = mini_study
    complete = study.get_trials(states=(TrialState.COMPLETE,))
    assert len(complete) == 2  # 1 in the first session + 1 after resuming, not 1 + 2
    assert complete[0].params != complete[1].params  # the resumed session drew anew
    for trial in complete:
        assert trial.user_attrs["epochs_run"] == trial.params["epochs"]
        assert trial.user_attrs["diverged"] is False
        assert trial.value == pytest.approx(trial.user_attrs["val_metrics"]["error"])
    assert study.user_attrs["method"] == "random" and study.user_attrs["seed"] == 0
    assert any(key.startswith("session_") for key in study.user_attrs)
    for name in ("study.db", "trials.csv", "best_config.yaml"):
        assert (study_dir / name).is_file(), name


def test_best_config_holds_the_best_parameters(mini_study):
    _, base, study, study_dir, _ = mini_study
    text = (study_dir / "best_config.yaml").read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    assert cfg["run_name"] == "best_random_seed0"
    for key, value in study.best_params.items():
        assert cfg["train"][key] == value
    assert cfg["model"] == base["model"] and cfg["split"] == base["split"]


def test_study_is_restored_from_the_backup(mini_study):
    hpo, base, _, study_dir, backup = mini_study
    assert (backup / "study.db").is_file() and (backup / "trials.csv").is_file()
    # A new machine (e.g. a fresh Colab VM): empty local folder, same backup folder.
    fresh_dir = study_dir.parent.parent / "fresh_vm" / study_dir.name
    study = run_study(hpo, base, "random", 0, fresh_dir, device="cpu", backup_dir=backup)
    assert (fresh_dir / "study.db").is_file()
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 2  # nothing re-run


def test_load_study_results(mini_study):
    _, _, study, study_dir, _ = mini_study
    results = load_study_results(study_dir / "study.db")
    assert (results["method"], results["seed"]) == ("random", 0)
    assert results["values"] == [t.value for t in study.get_trials(states=(TrialState.COMPLETE,))]
    assert len(results["params"]) == len(results["train_time_s"]) == 2
    assert results["best_number"] == study.best_trial.number


def test_load_study_results_without_completed_trials(tmp_path):
    # A study that has just started (only a running trial) must not break the comparison.
    db = tmp_path / "study.db"
    study = optuna.create_study(study_name="gp_seed1", storage=f"sqlite:///{db.as_posix()}")
    study.set_user_attr("method", "gp")
    study.set_user_attr("seed", 1)
    study.ask()
    results = load_study_results(db)
    assert results["values"] == [] and results["best_number"] is None


def _study_with_trials(db, trials, constant_error=0.25):
    """A finished study in ``db`` built from (value, user_attrs) pairs."""
    study = optuna.create_study(study_name="gp_seed0", storage=f"sqlite:///{db.as_posix()}")
    study.set_user_attr("method", "gp")
    study.set_user_attr("seed", 0)
    if constant_error is not None:
        study.set_user_attr("constant_predictor_error", constant_error)
    dist = {"lr": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True)}
    for value, attrs in trials:
        study.add_trial(
            optuna.trial.create_trial(
                value=value, params={"lr": 1e-3}, distributions=dist, user_attrs=attrs
            )
        )
    return study


def _metrics(recall, specificity):
    return {"recall": recall, "specificity": specificity}


def test_predicts_single_class():
    assert predicts_single_class(_metrics(0.0, 1.0))  # all NORMAL
    assert predicts_single_class(_metrics(1.0, 0.0))  # all PNEUMONIA
    assert not predicts_single_class(_metrics(1.0, 1.0))  # perfect
    assert not predicts_single_class(_metrics(0.99, 0.0))


def test_load_study_results_rescores_failed_trials(tmp_path):
    # Trials scored before the single-class rule (no "collapsed" attribute) are rescored
    # from their stored metrics; the stored objective values stay available.
    db = tmp_path / "study.db"
    _study_with_trials(
        db,
        [
            (0.75, {"diverged": False, "val_metrics": _metrics(0.0, 1.0)}),  # all NORMAL
            (0.25, {"diverged": True, "val_metrics": _metrics(0.0, 1.0)}),
            (0.01, {"diverged": False, "val_metrics": _metrics(0.99, 0.98)}),
            (0.25, {"diverged": False, "collapsed": True, "val_metrics": _metrics(1.0, 0.0)}),
        ],
    )
    results = load_study_results(db)
    assert results["raw_values"] == [0.75, 0.25, 0.01, 0.25]
    assert results["values"] == [0.25, 0.25, 0.01, 0.25]
    assert results["failed"] == [True, True, False, True]
    assert results["best_number"] == 2
    assert summarize([results])[0]["n_failed"] == 3


def test_load_study_results_needs_the_constant_error_for_failed_trials(tmp_path):
    db = tmp_path / "study.db"
    _study_with_trials(db, [(0.75, {"val_metrics": _metrics(0.0, 1.0)})], constant_error=None)
    with pytest.raises(ValueError, match="constant_predictor_error"):
        load_study_results(db)


def test_single_class_trial_is_scored_as_the_constant_predictor(data_root, tmp_path, monkeypatch):
    # A dead network outputs the same class for every image; which class depends on a
    # negligible bias difference, so it is scored like a diverged trial.
    def collapsed_fit(*args, **kwargs):
        val = {k: 0.5 for k in ("accuracy", "f1", "auc", "loss")}
        val.update(error=0.75, recall=0.0, specificity=1.0)
        return [{"epoch": 1, "train_nonfinite_fraction": 0.0}], (val,)

    monkeypatch.setattr("src.hpo.fit", collapsed_fit)
    hpo, base = _mini_configs(data_root, tmp_path)
    hpo["n_trials"] = 1
    study = run_study(hpo, base, "random", 0, tmp_path / "collapse", device="cpu")
    trial = study.trials[0]
    assert trial.user_attrs["collapsed"] is True and trial.user_attrs["diverged"] is False
    assert trial.user_attrs["val_metrics"]["error"] == 0.75  # measured value kept
    assert trial.value == pytest.approx(study.user_attrs["constant_predictor_error"])
    splits = build_splits(base, include_test=False)
    assert trial.value == pytest.approx(
        constant_predictor_error(splits["train"][1], splits["val"][1])
    )


def test_completed_old_study_gets_the_constant_error(data_root, tmp_path):
    # Rerunning hpo.py on a finished study from an older version adds the attribute
    # without training anything.
    hpo, base = _mini_configs(data_root, tmp_path)
    hpo["n_trials"] = 1
    study_dir = tmp_path / "old"
    study_dir.mkdir()
    db = study_dir / "study.db"
    _study_with_trials(db, [(0.75, {"val_metrics": _metrics(0.0, 1.0)})], constant_error=None)
    study = run_study(hpo, base, "gp", 0, study_dir, device="cpu")
    assert len(study.trials) == 1
    assert load_study_results(db)["values"] == [study.user_attrs["constant_predictor_error"]]


def test_diverging_trial_stops_early(data_root, tmp_path):
    hpo, base = _mini_configs(data_root, tmp_path)
    hpo["n_trials"] = 1
    hpo["search_space"] = {
        "lr": {"type": "float", "low": 1e6, "high": 2e6},  # guaranteed to blow up
        "epochs": {"type": "int", "low": 3, "high": 3},
    }
    study = run_study(hpo, base, "random", 0, tmp_path / "diverge", device="cpu")
    trial = study.trials[0]
    assert trial.state == TrialState.COMPLETE
    assert trial.user_attrs["diverged"] is True
    assert trial.user_attrs["epochs_run"] < 3
    assert trial.user_attrs["collapsed"] is False  # reported as diverged, not collapsed
    # Scored as the majority-class predictor, whatever the diverged model output.
    splits = build_splits(base, include_test=False)
    assert trial.value == pytest.approx(
        constant_predictor_error(splits["train"][1], splits["val"][1])
    )


def test_constant_predictor_error():
    # Majority training class is 1 -> every val image predicted 1 -> error = share of 0s.
    assert constant_predictor_error([0, 1, 1, 1], [0, 0, 1, 1, 1]) == pytest.approx(2 / 5)
    assert constant_predictor_error([0, 0, 1], [0, 1, 1, 1]) == pytest.approx(3 / 4)


def test_has_diverged():
    model = torch.nn.Linear(2, 2)
    healthy = [{"train_nonfinite_fraction": 0.0}]
    assert not has_diverged(model, healthy)
    # Some non-finite batches are tolerated; a whole epoch of them is divergence.
    assert not has_diverged(model, [{"train_nonfinite_fraction": 0.9}])
    assert has_diverged(model, [{"train_nonfinite_fraction": 1.0}])
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    assert has_diverged(model, healthy)


# --- analysis ----------------------------------------------------------------------------


def _fake_results():
    return [
        {"name": "gp_seed0", "method": "gp", "seed": 0, "values": [0.3, 0.02, 0.01, 0.012],
         "params": [{"lr": 1e-3 * (i + 1), "batch_size": 16} for i in range(4)],
         "train_time_s": [600.0] * 4, "best_number": 2},
        {"name": "random_seed0", "method": "random", "seed": 0, "values": [0.05, 0.04, 0.03],
         "params": [{"lr": 2e-3 * (i + 1), "batch_size": 32} for i in range(3)],
         "train_time_s": [300.0] * 3, "best_number": 2},
    ]  # fmt: skip


def test_best_so_far():
    np.testing.assert_array_equal(best_so_far([0.3, 0.1, 0.2, 0.05]), [0.3, 0.1, 0.1, 0.05])


def test_summarize():
    rows = summarize(_fake_results(), baseline_error=0.0115)
    gp, rnd = rows
    assert (gp["study"], gp["best_error"], gp["best_trial_index"]) == ("gp_seed0", 0.01, 3)
    assert gp["trials_to_beat_baseline"] == 3
    assert rnd["trials_to_beat_baseline"] is None
    assert gp["total_time_h"] == pytest.approx(4 * 600 / 3600, abs=0.01)
    assert gp["best_lr"] == pytest.approx(3e-3)
    assert gp["n_beat_baseline"] == 1 and rnd["n_beat_baseline"] == 0
    assert gp["mean_top5_error"] == pytest.approx(np.mean([0.3, 0.02, 0.01, 0.012]))
    assert "startup_best_error" not in gp


def test_summarize_splits_the_initial_design():
    gp, rnd = summarize(_fake_results(), n_startup_trials=2, top_k=2)
    assert gp["mean_top2_error"] == pytest.approx(0.011)
    assert (gp["startup_best_error"], gp["after_startup_best_error"]) == (0.02, 0.01)
    assert gp["improved_after_startup"]
    assert (rnd["startup_best_error"], rnd["after_startup_best_error"]) == (0.04, 0.03)
    # A study that has not left its initial design yet has no later trials.
    (only_startup,) = summarize(_fake_results()[1:], n_startup_trials=5)
    assert only_startup["after_startup_best_error"] is None
    assert not only_startup["improved_after_startup"]


def test_hpo_plots_are_written(tmp_path):
    results = _fake_results()
    space = {"lr": {"type": "float", "log": True}, "batch_size": {"type": "categorical"}}
    plot_best_so_far(results, 0.0115, tmp_path / "a.png", n_startup_trials=2)
    plot_trial_errors(results, 0.0115, tmp_path / "b.png")
    plot_params_vs_error(results, space, tmp_path / "c.png")
    assert all((tmp_path / name).stat().st_size > 0 for name in ("a.png", "b.png", "c.png"))
