"""Tests for src/model.py (random initialization only, no weight download)."""

import torch
from torch import nn

from src.model import build_alexnet


def test_build_alexnet_output_shape():
    model = build_alexnet(num_classes=2, pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 227, 227))
    assert logits.shape == (2, 2)


def test_build_alexnet_head_and_full_fine_tuning():
    model = build_alexnet(num_classes=2, pretrained=False)
    head = model.classifier[6]
    assert isinstance(head, nn.Linear)
    assert (head.in_features, head.out_features) == (4096, 2)
    assert all(p.requires_grad for p in model.parameters())
