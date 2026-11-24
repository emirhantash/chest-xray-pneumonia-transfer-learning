"""Training and evaluation runs: setup, resumable epoch loop, and output files.

Run folder layout (``<output_dir>/<run_name>/``):
    config.yaml          effective configuration of the run
    run_info.json        library versions, device, split fingerprint
    history.csv          one row per epoch (train and validation metrics)
    last.pt              checkpoint after the latest completed epoch (resume point;
                         after the last epoch, this is the final model)
    val_metrics.json     validation metrics of the final model
    val_predictions.csv  per-image validation predictions of the final model
    curves.png, confusion_val.png, roc_val.png
    eval_<split>/        outputs of :func:`evaluate_run`
"""

import copy
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
import yaml

from src.checkpoint import get_rng_state, load_checkpoint, save_checkpoint, set_rng_state
from src.data import build_dataloaders, build_splits, split_fingerprint
from src.engine import build_criterion, build_optimizer, evaluate, train_one_epoch, use_amp
from src.metrics import roc_points
from src.model import build_alexnet
from src.plots import plot_confusion_matrix, plot_history, plot_roc
from src.utils import get_device, git_revision, set_seed

CHECKPOINT_NAME = "last.pt"
# Validation metrics logged per epoch (prefixed with "val_" in history.csv).
VAL_LOG_KEYS = [
    "loss",
    "accuracy",
    "error",
    "recall",
    "specificity",
    "precision",
    "f1",
    "balanced_accuracy",
    "auc",
]
HISTORY_FIELDS = (
    ["epoch", "lr", "train_loss", "train_accuracy", "train_nonfinite_fraction"]
    + [f"val_{k}" for k in VAL_LOG_KEYS]
    + ["epoch_time_s"]
)
# Machine-specific settings that may differ when a run is resumed or evaluated elsewhere.
MACHINE_KEYS = ("root", "num_workers")


def _write_json(path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_history(path, history: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def _write_predictions(path, paths, root, y_true, y_prob, y_logit, threshold: float) -> None:
    """One row per image; ``logit_pneumonia`` (log-odds) keeps the ranking of confident
    predictions whose probability rounds to 1.000000."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "prob_pneumonia", "logit_pneumonia", "pred"])
        for p, y, prob, logit in zip(paths, y_true, y_prob, y_logit, strict=True):
            rel = Path(p).relative_to(root).as_posix()
            writer.writerow([rel, int(y), f"{prob:.6f}", f"{logit:.9g}", int(prob >= threshold)])


def _save_eval_outputs(out_dir, prefix, metrics, paths, root, y_true, y_prob, y_logit, threshold):
    """Predictions CSV, confusion matrix, and ROC curve for one evaluated subset."""
    _write_predictions(
        out_dir / f"{prefix}_predictions.csv", paths, root, y_true, y_prob, y_logit, threshold
    )
    plot_confusion_matrix(
        metrics, out_dir / f"confusion_{prefix}.png", f"Confusion matrix ({prefix})"
    )
    fpr, tpr = roc_points(y_true, y_logit)
    plot_roc(fpr, tpr, metrics["auc"], out_dir / f"roc_{prefix}.png", f"ROC ({prefix})")


def _with_machine_settings(saved_cfg: dict, cfg: dict) -> dict:
    """The saved run config, with the data location and worker count of this machine."""
    merged = copy.deepcopy(saved_cfg)
    for key in MACHINE_KEYS:
        merged["data"][key] = cfg["data"][key]
    return merged


def _run_info(cfg, device, amp: bool, fingerprint: str) -> dict:
    return {
        "started": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": amp,
        "split_source": cfg["split"]["file"] or "computed",
        "split_fingerprint": fingerprint,
        "git": git_revision(),
    }


def format_epoch(row: dict, n_epochs: int) -> str:
    """One log line for a history row."""
    return (
        f"epoch {row['epoch']:2d}/{n_epochs}  train loss {row['train_loss']:.4f} "
        f"acc {row['train_accuracy']:.4f} | val loss {row['val_loss']:.4f} "
        f"acc {row['val_accuracy']:.4f} recall {row['val_recall']:.4f} "
        f"spec {row['val_specificity']:.4f} auc {row['val_auc']:.4f} | "
        f"{row['epoch_time_s']:.1f} s"
    )


def fit(
    model,
    loaders: dict,
    criterion,
    optimizer,
    scaler,
    generator: torch.Generator,
    cfg: dict,
    device: torch.device,
    amp: bool,
    history: list[dict] | None = None,
    start_epoch: int = 1,
    on_epoch_end=None,
):
    """Train epochs ``start_epoch..train.epochs``, evaluating on val after each epoch.

    ``on_epoch_end(epoch, history)`` runs after every epoch; returning True stops the
    loop (used to simulate an interruption, and by HPO trials to stop once the weights
    have become NaN/inf).

    Returns ``(history, latest)``; ``latest = (val_metrics, y_true, y_prob)`` of the
    last epoch run in this call, or None if no epoch was left.
    """
    history = [] if history is None else history
    seed, n_epochs = cfg["seed"], cfg["train"]["epochs"]
    threshold = cfg["eval"]["threshold"]
    latest = None
    for epoch in range(start_epoch, n_epochs + 1):
        # Re-seeding the shuffling generator per epoch makes the batch order of an epoch
        # independent of how many epochs ran in the current session (exact resume).
        generator.manual_seed(seed + epoch)
        start = time.perf_counter()
        train_stats = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, amp
        )
        latest = evaluate(model, loaders["val"], criterion, device, threshold)
        val = latest[0]
        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "train_nonfinite_fraction": train_stats["nonfinite_fraction"],
                **{f"val_{k}": val[k] for k in VAL_LOG_KEYS},
                "epoch_time_s": round(time.perf_counter() - start, 2),
            }
        )
        print(format_epoch(history[-1], n_epochs), flush=True)
        if on_epoch_end is not None and on_epoch_end(epoch, history):
            break
    return history, latest


def run_training(
    cfg: dict,
    run_dir,
    resume: bool = False,
    device: str | None = None,
    stop_after: int | None = None,
) -> dict:
    """Train for ``train.epochs`` epochs on train, evaluating on val after every epoch.

    The model after the final epoch is the result of the run (no early stopping, no
    checkpoint selection). A checkpoint is written after every epoch; with
    ``resume=True`` training continues from it with the *saved* config (only the data
    location and worker count are taken from ``cfg``). ``stop_after`` ends the session
    after that epoch, as if it had been interrupted (used in tests).

    The original test set is never loaded here.

    Returns a summary dict with the last completed epoch and the latest val metrics.
    """
    run_dir = Path(run_dir)
    ckpt_path = run_dir / CHECKPOINT_NAME
    ckpt = None
    if resume:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"No checkpoint to resume from: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path)
        cfg = _with_machine_settings(ckpt["config"], cfg)
    elif ckpt_path.exists():
        raise FileExistsError(
            f"{ckpt_path} already exists. Resume it (--resume) or use another --run-name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device) if device else get_device()
    amp = use_amp(cfg, device)
    threshold = cfg["eval"]["threshold"]
    seed = cfg["seed"]
    generator = set_seed(seed)

    root = Path(cfg["data"]["root"])
    splits = build_splits(cfg, include_test=False)
    fingerprint = split_fingerprint(splits, root)
    if ckpt is not None and ckpt["split_fingerprint"] != fingerprint:
        raise ValueError("The train/val split differs from the one the checkpoint was trained on.")
    loaders = build_dataloaders(splits, cfg, generator)

    # On resume the weights come from the checkpoint, so skip the ImageNet download.
    pretrained = cfg["model"]["pretrained"] and ckpt is None
    model = build_alexnet(cfg["model"]["num_classes"], pretrained=pretrained).to(device)
    criterion = build_criterion(splits["train"][1], cfg, device)
    optimizer = build_optimizer(model, cfg["train"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    if ckpt is None:
        history, start_epoch = [], 1
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        _write_json(run_dir / "run_info.json", _run_info(cfg, device, amp, fingerprint))
    else:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if amp and ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
        history, start_epoch = ckpt["history"], ckpt["epoch"] + 1
        set_rng_state(ckpt["rng"])
        print(f"Resuming {run_dir} after epoch {ckpt['epoch']}")

    n_epochs = cfg["train"]["epochs"]
    epochs_left = (
        f"epochs {start_epoch}..{n_epochs}"
        if start_epoch <= n_epochs
        else "already complete, regenerating the final outputs"
    )
    print(
        f"Training on {device} (amp={amp}): {len(splits['train'][0])} train / "
        f"{len(splits['val'][0])} val images, {epochs_left}"
    )

    def on_epoch_end(epoch: int, history: list[dict]) -> bool:
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "rng": get_rng_state(),
                "config": cfg,
                "split_fingerprint": fingerprint,
            },
            ckpt_path,
        )
        _write_history(run_dir / "history.csv", history)
        return stop_after is not None and epoch >= stop_after

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
        history=history,
        start_epoch=start_epoch,
        on_epoch_end=on_epoch_end,
    )
    if history[-1]["epoch"] < n_epochs:  # stopped early by ``stop_after``
        return {"epoch": history[-1]["epoch"], "completed": False, "val_metrics": latest[0]}

    if latest is None:  # the run was already complete: recompute the final predictions
        latest = evaluate(model, loaders["val"], criterion, device, threshold)
    val, y_true, y_prob, y_logit = latest
    best = min(history, key=lambda row: row["val_error"])
    _write_json(
        run_dir / "val_metrics.json",
        {
            "selection": "final epoch (no early stopping)",
            "epoch": history[-1]["epoch"],
            "threshold": threshold,
            "metrics": val,
            # For information only; the final-epoch model is the result of the run.
            "lowest_val_error_epoch": best["epoch"],
            "lowest_val_error": best["val_error"],
        },
    )
    _save_eval_outputs(
        run_dir, "val", val, splits["val"][0], root, y_true, y_prob, y_logit, threshold
    )
    plot_history(history, run_dir / "curves.png")
    return {"epoch": history[-1]["epoch"], "completed": True, "val_metrics": val}


def evaluate_run(
    run_dir,
    split: str = "val",
    final: bool = False,
    data_root: str | None = None,
    device: str | None = None,
) -> dict:
    """Evaluate the final model of a completed run on ``val`` or ``test``.

    The original test set is reserved for the final evaluation (Phase 4): ``split="test"``
    requires ``final=True``. Outputs go to ``<run_dir>/eval_<split>/``.
    """
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    if split == "test" and not final:
        raise ValueError(
            "The test set is reserved for the final evaluation; pass final=True (--final)."
        )
    run_dir = Path(run_dir)
    ckpt = load_checkpoint(run_dir / CHECKPOINT_NAME)
    cfg = copy.deepcopy(ckpt["config"])
    if data_root:
        cfg["data"]["root"] = data_root
    if ckpt["epoch"] < cfg["train"]["epochs"]:
        raise RuntimeError(
            f"Run {run_dir} stopped after epoch {ckpt['epoch']} of {cfg['train']['epochs']}; "
            "resume it before evaluating."
        )

    device = torch.device(device) if device else get_device()
    threshold = cfg["eval"]["threshold"]
    generator = set_seed(cfg["seed"])
    root = Path(cfg["data"]["root"])
    splits = build_splits(cfg, include_test=(split == "test"))
    if split_fingerprint(splits, root) != ckpt["split_fingerprint"]:
        raise ValueError("The train/val split differs from the one the run was trained on.")
    loader = build_dataloaders({split: splits[split]}, cfg, generator)[split]

    model = build_alexnet(cfg["model"]["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    criterion = build_criterion(splits["train"][1], cfg, device)
    metrics, y_true, y_prob, y_logit = evaluate(model, loader, criterion, device, threshold)

    out_dir = run_dir / f"eval_{split}"
    out_dir.mkdir(exist_ok=True)
    _write_json(
        out_dir / "metrics.json",
        {"split": split, "epoch": ckpt["epoch"], "threshold": threshold, "metrics": metrics},
    )
    _save_eval_outputs(
        out_dir, split, metrics, splits[split][0], root, y_true, y_prob, y_logit, threshold
    )
    return metrics
