import torch
import torch.nn as nn
from torchvision.models import resnet18


def get_model(num_classes: int = 100) -> nn.Module:
    """ResNet18 for CIFAR-100 (100 classes)."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")