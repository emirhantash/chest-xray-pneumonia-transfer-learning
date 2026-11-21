"""Figures: learning curves, confusion matrix, ROC curve.

Uses matplotlib's object-oriented ``Figure`` API (no pyplot), so no GUI backend is
needed and the functions are safe to call from scripts, notebooks, and tests.
"""

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
