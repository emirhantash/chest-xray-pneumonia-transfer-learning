"""Optimizer, loss, and the per-epoch training and evaluation loops."""

import numpy as np
import torch
from torch import nn

from src.data import compute_class_weights
from src.metrics import binary_metrics


def use_amp(cfg, device: torch.device) -> bool:
    """Mixed precision only on CUDA (``train.amp``); MPS and CPU always use float32."""
    return bool(cfg["train"]["amp"]) and device.type == "cuda"


def build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    """SGD with momentum and L2 weight decay on all parameters (interim report, Table 3)."""
    return torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["lr"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )


def build_criterion(train_labels, cfg, device: torch.device) -> nn.CrossEntropyLoss:
    """Cross-entropy, with inverse-frequency class weights from the training subset."""
    weight = None
    if cfg["train"]["use_class_weights"]:
        weight = compute_class_weights(train_labels, cfg["model"]["num_classes"]).to(device)
    return nn.CrossEntropyLoss(weight=weight)


def _batch_weight(criterion: nn.CrossEntropyLoss, labels: torch.Tensor) -> torch.Tensor:
    """Denominator of the batch's (weighted) mean loss, to average exactly over an epoch."""
    if criterion.weight is None:
        return torch.tensor(float(labels.numel()), device=labels.device)
    return criterion.weight[labels].sum()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp: bool) -> dict:
    """One pass over ``loader``; returns the epoch's mean loss and running accuracy.

    Both are measured in training mode (dropout active) while the weights change, as is
    usual for training curves; validation metrics come from :func:`evaluate`.
    """
    model.train()
    loss_sum = torch.zeros((), device=device)
    weight_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    n_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_weight = _batch_weight(criterion, labels)
        loss_sum += loss.detach().float() * batch_weight
        weight_sum += batch_weight
        correct += (logits.argmax(dim=1) == labels).sum()
        n_samples += labels.numel()
    return {"loss": (loss_sum / weight_sum).item(), "accuracy": correct.item() / n_samples}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp: bool, threshold: float):
    """Evaluate in eval mode; returns ``(metrics, y_true, y_prob)``.

    ``y_prob`` is the softmax probability of PNEUMONIA; ``metrics`` adds the epoch's
    mean (weighted) loss to :func:`src.metrics.binary_metrics`.
    """
    model.eval()
    loss_sum = torch.zeros((), device=device)
    weight_sum = torch.zeros((), device=device)
    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        logits = logits.float()
        batch_weight = _batch_weight(criterion, labels)
        loss_sum += criterion(logits, labels) * batch_weight
        weight_sum += batch_weight
        all_labels.append(labels.cpu())
        all_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy().astype(np.float64)
    metrics = binary_metrics(y_true, y_prob, threshold)
    metrics["loss"] = (loss_sum / weight_sum).item()
    return metrics, y_true, y_prob
