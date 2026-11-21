"""Figures: learning curves, confusion matrix, ROC curve, and HPO comparison plots.

Uses matplotlib's object-oriented ``Figure`` API (no pyplot), so no GUI backend is
needed and the functions are safe to call from scripts, notebooks, and tests.
"""

import numpy as np
from matplotlib.figure import Figure

from src.data import CLASSES


def plot_history(history: list[dict], path) -> None:
    """Loss, accuracy, and validation recall/specificity per epoch."""
    epochs = [row["epoch"] for row in history]
    fig = Figure(figsize=(15, 4), layout="constrained")
    ax_loss, ax_acc, ax_rates = fig.subplots(1, 3)

    ax_loss.plot(epochs, [r["train_loss"] for r in history], "o-", label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], "o-", label="val")
    ax_loss.set(title="Weighted cross-entropy loss", xlabel="epoch", ylabel="loss")

    ax_acc.plot(epochs, [r["train_accuracy"] for r in history], "o-", label="train (running)")
    ax_acc.plot(epochs, [r["val_accuracy"] for r in history], "o-", label="val")
    ax_acc.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")

    ax_rates.plot(epochs, [r["val_recall"] for r in history], "o-", label="recall")
    ax_rates.plot(epochs, [r["val_specificity"] for r in history], "o-", label="specificity")
    ax_rates.plot(epochs, [r["val_f1"] for r in history], "o-", label="F1")
    ax_rates.set(title="Validation recall / specificity / F1", xlabel="epoch")

    for ax in (ax_loss, ax_acc, ax_rates):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(path, dpi=120)


def plot_confusion_matrix(metrics: dict, path, title: str) -> None:
    """2x2 confusion matrix (rows: true class, columns: predicted class)."""
    matrix = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    fig = Figure(figsize=(4.5, 4), layout="constrained")
    ax = fig.subplots()
    ax.imshow(matrix, cmap="Blues")
    peak = max(max(row) for row in matrix)
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i][j] > peak / 2 else "black"
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color=color, fontsize=14)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="predicted",
        ylabel="true",
        title=title,
    )
    fig.savefig(path, dpi=120)


def plot_roc(fpr, tpr, auc: float, path, title: str) -> None:
    """ROC curve with the chance diagonal."""
    fig = Figure(figsize=(4.5, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    ax.set(
        xlabel="false positive rate (1 - specificity)",
        ylabel="true positive rate (recall)",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=120)


# --- hyperparameter optimization ---------------------------------------------------------

METHOD_COLORS = {"gp": "C0", "random": "C1"}
METHOD_LABELS = {"gp": "Bayesian optimization (GP)", "random": "Random Search"}
# Plotted errors are clipped here so that log axes also work for a perfect trial.
MIN_PLOTTED_ERROR = 1e-4


def _by_method(results: list[dict]) -> dict:
    grouped = {}
    for r in sorted(results, key=lambda r: (r["method"], r["seed"])):
        grouped.setdefault(r["method"], []).append(r)
    return grouped


def _baseline_line(ax, baseline_error):
    if baseline_error is not None:
        ax.axhline(baseline_error, color="gray", ls="--", label=f"baseline ({baseline_error:.4f})")


def plot_best_so_far(results: list[dict], baseline_error, path) -> None:
    """Best validation error so far vs completed trials: each seed (thin) and the mean
    over seeds with its min-max band (thick), per method."""
    fig = Figure(figsize=(7, 4.5), layout="constrained")
    ax = fig.subplots()
    for method, runs in _by_method(results).items():
        color = METHOD_COLORS.get(method)
        curves = [np.minimum.accumulate(np.asarray(r["values"], dtype=float)) for r in runs]
        for curve in curves:
            ax.plot(np.arange(1, len(curve) + 1), curve, color=color, alpha=0.3, lw=1)
        n = min(len(c) for c in curves)
        stacked = np.stack([c[:n] for c in curves])
        x = np.arange(1, n + 1)
        label = f"{METHOD_LABELS.get(method, method)} (mean of {len(curves)} seeds)"
        ax.plot(x, stacked.mean(axis=0), color=color, lw=2.5, label=label)
        ax.fill_between(x, stacked.min(axis=0), stacked.max(axis=0), color=color, alpha=0.15)
    _baseline_line(ax, baseline_error)
    finite_best = [min(r["values"]) for r in results]
    if baseline_error is not None and finite_best:
        ax.set_ylim(0.8 * min(finite_best + [baseline_error]), 3 * baseline_error)
    ax.set(xlabel="completed trials", ylabel="best validation error so far")
    ax.set_title("HPO convergence")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=120)


def plot_trial_errors(results: list[dict], baseline_error, path) -> None:
    """Validation error of every completed trial (all seeds), log scale."""
    fig = Figure(figsize=(7, 4.5), layout="constrained")
    ax = fig.subplots()
    for method, runs in _by_method(results).items():
        xs = [i + 1 for r in runs for i in range(len(r["values"]))]
        ys = [max(v, MIN_PLOTTED_ERROR) for r in runs for v in r["values"]]
        ax.scatter(
            xs,
            ys,
            s=18,
            alpha=0.6,
            color=METHOD_COLORS.get(method),
            label=METHOD_LABELS.get(method, method),
        )
    _baseline_line(ax, baseline_error)
    ax.set_yscale("log")
    ax.set(xlabel="trial (completed, per study)", ylabel="validation error")
    ax.set_title("All trials")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(path, dpi=120)


def plot_params_vs_error(results: list[dict], space: dict, path) -> None:
    """Each searched hyperparameter vs validation error, colored by method."""
    names = list(space)
    fig = Figure(figsize=(4 * len(names), 3.8), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(names), sharey=True))
    for ax, name in zip(axes, names, strict=True):
        for method, runs in _by_method(results).items():
            xs = [p[name] for r in runs for p in r["params"]]
            ys = [max(v, MIN_PLOTTED_ERROR) for r in runs for v in r["values"]]
            ax.scatter(
                xs,
                ys,
                s=16,
                alpha=0.6,
                color=METHOD_COLORS.get(method),
                label=METHOD_LABELS.get(method, method),
            )
        if space[name].get("log"):
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("validation error")
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=120)
