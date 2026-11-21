"""Model definition."""

from torch import nn
from torchvision.models import AlexNet_Weights, alexnet


def build_alexnet(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """torchvision AlexNet with the last Linear layer replaced by a ``num_classes`` head.

    All layers stay trainable (full fine-tuning). The head outputs logits;
    CrossEntropyLoss applies the softmax internally.
    """
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = alexnet(weights=weights)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
