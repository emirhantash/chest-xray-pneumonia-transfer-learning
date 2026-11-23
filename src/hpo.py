"""Hyperparameter optimization with Optuna: GP-based Bayesian optimization vs Random Search.

One study = one (method, sampler seed) pair, stored in its own SQLite file
``<output_dir>/<method>_seed<seed>/study.db`` so that studies can run on different
machines. A study is resumable at trial granularity: rerunning it continues until
``n_trials`` trials have completed. A trial interrupted by a crash or a lost notebook
session is marked as failed and its parameters are queued again, so the interruption
does not change which configurations are evaluated.

Every trial trains with the base config's seed (same initialization and batch order for
equal batch sizes) and returns the validation error of its final-epoch model. A trial
that diverges (see :func:`has_diverged`) stops early and returns the error of the
trivial classifier that predicts the majority training class for every image, so a
failed run gets the same value on every device and with or without mixed precision.
The original test set is never loaded.
"""

import copy
import csv
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml
from optuna.samplers import GPSampler, RandomSampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from src.data import build_datasets, build_splits, make_loader, split_fingerprint
from src.engine import build_criterion, build_optimizer, use_amp
from src.model import build_alexnet
from src.runner import fit
from src.utils import get_device, git_revision, set_seed

METHODS = ("gp", "random")
STUDY_DB = "study.db"
# Validation metrics stored with every trial (the objective is "error").
TRIAL_METRICS = ("error", "accuracy", "recall", "specificity", "f1", "auc", "loss")


def study_name(method: str, seed: int) -> str:
    return f"{method}_seed{seed}"


def make_sampler(method: str, seed: int, n_startup_trials: int) -> optuna.samplers.BaseSampler:
    """``gp``: Gaussian-process Bayesian optimization; ``random``: Random Search."""
    if method == "gp":
        return GPSampler(seed=seed, n_startup_trials=n_startup_trials)
    if method == "random":
        return RandomSampler(seed=seed)
    raise ValueError(f"Unknown method {method!r} (expected one of {METHODS})")


def session_seed(seed: int, n_existing_trials: int) -> int:
    """Sampler seed for one optimization session.

    Samplers keep their random state in memory only. Deriving the seed from the number of
    trials already in the study keeps a resumed study from repeating the first session's
    draws; an uninterrupted study is fully determined by ``seed``.
    """
    return int(np.random.SeedSequence([seed, n_existing_trials]).generate_state(1)[0])


def suggest_params(trial: optuna.Trial, space: dict) -> dict:
    """Sample one configuration from the search space defined in the HPO config."""
    params = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown parameter type {kind!r} for {name!r}")
    return params


def apply_params(base_cfg: dict, params: dict) -> dict:
    """A copy of the base config with the searched values in its ``train`` section."""
    cfg = copy.deepcopy(base_cfg)
    unknown = params.keys() - cfg["train"].keys()
    if unknown:
        raise KeyError(f"Searched parameters not in the train config: {sorted(unknown)}")
    cfg["train"].update(params)
    return cfg


def has_diverged(model: torch.nn.Module, history: list[dict]) -> bool:
    """True if training cannot recover: NaN/inf weights, or a whole epoch without a
    single finite loss (with AMP, the gradient scaler then skips every step and the
    model is frozen; in float32 the weights become NaN)."""
    weights_finite = all(torch.isfinite(p).all() for p in model.parameters())
    return not weights_finite or history[-1]["train_nonfinite_fraction"] == 1.0


def constant_predictor_error(train_labels, val_labels) -> float:
    """Validation error of always predicting the most frequent training class."""
    majority = int(np.bincount(train_labels).argmax())
    return float(np.mean(np.asarray(val_labels) != majority))


class TrialObjective:
    """Optuna objective. Builds the datasets (and the image cache) once per process."""

    def __init__(self, base_cfg: dict, space: dict, device: torch.device):
        self.base_cfg = base_cfg
        self.space = space
        self.device = device
        splits = build_splits(base_cfg, include_test=False)
        self.fingerprint = split_fingerprint(splits, Path(base_cfg["data"]["root"]))
        self.train_labels = splits["train"][1]
        self.diverged_error = constant_predictor_error(splits["train"][1], splits["val"][1])
        self.datasets = build_datasets(splits, base_cfg)

    def __call__(self, trial: optuna.Trial) -> float:
        params = suggest_params(trial, self.space)
        cfg = apply_params(self.base_cfg, params)
        device, amp = self.device, use_amp(cfg, self.device)
        print(f"\nTrial {trial.number}: {params}", flush=True)

        generator = set_seed(cfg["seed"])
        loaders = {
            "train": make_loader(self.datasets["train"], True, cfg, generator),
            "val": make_loader(self.datasets["val"], False, cfg, generator),
        }
        model = build_alexnet(cfg["model"]["num_classes"], cfg["model"]["pretrained"]).to(device)
        criterion = build_criterion(self.train_labels, cfg, device)
        optimizer = build_optimizer(model, cfg["train"])
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        start = time.perf_counter()
        history, latest = fit(
            model,
            loaders,
            criterion,
            optimizer,
            scaler,
            generator,
            cfg,
            device,
            amp,
            on_epoch_end=lambda epoch, hist: has_diverged(model, hist),
        )
        val = latest[0]
        diverged = has_diverged(model, history)
        trial.set_user_attr("val_metrics", {k: float(val[k]) for k in TRIAL_METRICS})
        trial.set_user_attr("epochs_run", history[-1]["epoch"])
        trial.set_user_attr("diverged", diverged)
        trial.set_user_attr("train_time_s", round(time.perf_counter() - start, 1))
        # The model is freed when this returns; the CUDA caching allocator reuses its
        # memory for the next trial.
        if diverged:
            print(
                f"Trial {trial.number}: training diverged after epoch {history[-1]['epoch']}; "
                f"stopped, scored as the constant predictor ({self.diverged_error:.4f})."
            )
            return self.diverged_error
        return val["error"]


def mark_interrupted_trials(study: optuna.Study) -> list[int]:
    """Mark trials left RUNNING by a crashed session as FAIL (one process per study)."""
    numbers = [t.number for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))]
    for number in numbers:
        study.tell(number, state=TrialState.FAIL)
    return numbers


def requeue_failed_trials(study: optuna.Study) -> list[int]:
    """Queue the parameters of every failed trial again (once per failed trial)."""
    trials = study.get_trials(deepcopy=False)
    retried = {t.user_attrs.get("retry_of") for t in trials}
    requeued = []
    for t in trials:
        if t.state == TrialState.FAIL and t.params and t.number not in retried:
            study.enqueue_trial(t.params, user_attrs={"retry_of": t.number})
            requeued.append(t.number)
    return requeued


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def write_trials_csv(study: optuna.Study, path) -> None:
    """Human-readable table of all trials (state, objective, parameters, metrics)."""
    trials = study.get_trials(deepcopy=False)
    param_names = sorted({name for t in trials for name in t.params})
    fields = ["number", "state", "value", *param_names, "epochs_run", "diverged"]
    fields += [f"val_{k}" for k in TRIAL_METRICS if k != "error"]
    fields += ["train_time_s", "retry_of"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in trials:
            metrics = t.user_attrs.get("val_metrics", {})
            writer.writerow(
                {
                    "number": t.number,
                    "state": t.state.name,
                    "value": t.value,
                    **t.params,
                    "epochs_run": t.user_attrs.get("epochs_run"),
                    "diverged": t.user_attrs.get("diverged"),
                    **{f"val_{k}": metrics.get(k) for k in TRIAL_METRICS if k != "error"},
                    "train_time_s": t.user_attrs.get("train_time_s"),
                    "retry_of": t.user_attrs.get("retry_of"),
                }
            )


def export_best_config(study: optuna.Study, base_cfg: dict, path, run_name: str) -> dict:
    """Write the base config with the best trial's parameters (for retraining)."""
    cfg = apply_params(base_cfg, study.best_trial.params)
    cfg["run_name"] = run_name
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# Best configuration of study {study.study_name} (trial "
            f"{study.best_trial.number}, val error {study.best_value:.6f}).\n"
        )
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def _check_study_settings(study: optuna.Study, expected: dict) -> None:
    """Refuse to resume a study with a different method, seed, or search space."""
    for key, value in expected.items():
        stored = study.user_attrs.get(key)
        if stored is not None and stored != value:
            raise ValueError(
                f"Study {study.study_name} was created with {key}={stored!r}, not {value!r}."
            )


def run_study(
    hpo_cfg: dict,
    base_cfg: dict,
    method: str,
    seed: int,
    study_dir,
    n_trials: int | None = None,
    device: str | None = None,
    backup_dir=None,
) -> optuna.Study:
    """Create or resume one study and run it until ``n_trials`` trials are complete.

    ``backup_dir`` (e.g. a Google Drive folder): the SQLite file is used on local disk
    and copied there atomically after every trial (SQLite on network/FUSE file systems
    is unreliable); a missing local file is restored from the backup.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r} (expected one of {METHODS})")
    n_trials = n_trials or hpo_cfg["n_trials"]
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / STUDY_DB
    backup_path = Path(backup_dir) / STUDY_DB if backup_dir else None
    if backup_path is not None and backup_path.exists() and not db_path.exists():
        shutil.copy2(backup_path, db_path)
        print(f"Restored {db_path} from {backup_path}")

    storage = f"sqlite:///{db_path.resolve().as_posix()}"
    name = study_name(method, seed)
    settings = {
        "method": method,
        "seed": seed,
        "n_startup_trials": hpo_cfg["n_startup_trials"],
        "search_space": hpo_cfg["search_space"],
    }
    study = optuna.create_study(
        storage=storage, study_name=name, direction="minimize", load_if_exists=True
    )
    _check_study_settings(study, settings)

    interrupted = mark_interrupted_trials(study)
    requeued = requeue_failed_trials(study)
    if interrupted or requeued:
        print(f"Interrupted trials marked as failed: {interrupted}; re-queued: {requeued}")

    n_complete = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    print(f"Study {name}: {n_complete}/{n_trials} trials complete ({db_path})")
    if n_complete < n_trials:
        device = torch.device(device) if device else get_device()
        objective = TrialObjective(base_cfg, hpo_cfg["search_space"], device)
        if not study.user_attrs:
            for key, value in settings.items():
                study.set_user_attr(key, value)
            study.set_user_attr("split_fingerprint", objective.fingerprint)
        if study.user_attrs.get("split_fingerprint") != objective.fingerprint:
            raise ValueError(f"Study {name} was run on a different train/val split.")
        study.set_user_attr(f"session_{len(study.trials)}", _session_info(device))

        study.sampler = make_sampler(
            method, session_seed(seed, len(study.trials)), hpo_cfg["n_startup_trials"]
        )

        def after_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            write_trials_csv(study, study_dir / "trials.csv")
            if backup_path is not None:
                _atomic_copy(db_path, backup_path)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,)), after_trial],
            gc_after_trial=True,
        )

    write_trials_csv(study, study_dir / "trials.csv")
    export_best_config(study, base_cfg, study_dir / "best_config.yaml", f"best_{name}")
    if backup_path is not None:
        _atomic_copy(db_path, backup_path)
        _atomic_copy(study_dir / "trials.csv", backup_path.parent / "trials.csv")
        _atomic_copy(study_dir / "best_config.yaml", backup_path.parent / "best_config.yaml")
    return study


def _session_info(device: torch.device) -> dict:
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "optuna": optuna.__version__,
        "git": git_revision(),
    }


# --- analysis ----------------------------------------------------------------------------


def load_study_results(db_path) -> dict:
    """Completed trials of the single study in ``db_path``, in trial order."""
    storage = f"sqlite:///{Path(db_path).resolve().as_posix()}"
    (name,) = optuna.get_all_study_names(storage)
    study = optuna.load_study(study_name=name, storage=storage)
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return {
        "name": name,
        "method": study.user_attrs["method"],
        "seed": study.user_attrs["seed"],
        "values": [t.value for t in trials],
        "params": [t.params for t in trials],
        "train_time_s": [t.user_attrs.get("train_time_s", math.nan) for t in trials],
        "best_number": study.best_trial.number,
    }


def best_so_far(values) -> np.ndarray:
    """Running minimum of the objective over completed trials."""
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def summarize(
    results: list[dict],
    baseline_error: float | None = None,
    n_startup_trials: int | None = None,
    top_k: int = 5,
) -> list[dict]:
    """One summary row per study, sorted by method and seed.

    Besides the best error, it reports measures that still separate the methods when
    many trials tie at the best value on a saturated validation set: the mean of the
    ``top_k`` lowest errors and the number of trials better than the baseline. With
    ``n_startup_trials``, it also splits the study into the random initial design and
    the trials after it (``GPSampler`` and ``RandomSampler`` with the same seed draw the
    same initial design, so only the later trials show what the GP adds).
    """
    rows = []
    for r in sorted(results, key=lambda r: (r["method"], r["seed"])):
        values = np.asarray(r["values"], dtype=float)
        best_idx = int(np.argmin(values))
        row = {
            "study": r["name"],
            "method": r["method"],
            "seed": r["seed"],
            "n_trials": len(values),
            "best_error": float(values[best_idx]),
            "best_trial_index": best_idx + 1,  # 1-based position among completed trials
            "median_error": float(np.median(values)),
            f"mean_top{top_k}_error": float(np.sort(values)[:top_k].mean()),
            "total_time_h": round(float(np.nansum(r["train_time_s"])) / 3600, 2),
            **{f"best_{k}": v for k, v in r["params"][best_idx].items()},
        }
        if baseline_error is not None:
            better = np.flatnonzero(values < baseline_error)
            row["trials_to_beat_baseline"] = int(better[0]) + 1 if better.size else None
            row["n_beat_baseline"] = int(better.size)
        if n_startup_trials is not None:
            startup, later = values[:n_startup_trials], values[n_startup_trials:]
            row["startup_best_error"] = float(startup.min())
            row["after_startup_best_error"] = float(later.min()) if later.size else None
            row["improved_after_startup"] = bool(later.size and later.min() < startup.min())
        rows.append(row)
    return rows
