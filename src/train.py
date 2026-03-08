from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_local(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    local_epochs: int = 5,
    lr: float = 0.03,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    label_smoothing: float = 0.05,
) -> Dict[str, float | List[float]]:
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=local_epochs,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epoch_losses: List[float] = []

    for _epoch in range(local_epochs):
        total_loss = 0.0
        total_samples = 0

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        scheduler.step()

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_losses.append(epoch_loss)

    return {
        "train_loss_last": float(epoch_losses[-1]),
        "train_loss_mean": float(sum(epoch_losses) / len(epoch_losses)),
        "epoch_losses": epoch_losses,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 100,
) -> Dict[str, float]:
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        labels_cpu = labels.cpu()
        preds_cpu = preds.cpu()
        idx = labels_cpu * num_classes + preds_cpu
        binc = torch.bincount(idx, minlength=num_classes * num_classes)
        confusion += binc.reshape(num_classes, num_classes)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)

    tp = confusion.diag().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp
    support = confusion.sum(dim=1).float()

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    f1_macro = f1.mean().item()

    if support.sum() > 0:
        f1_weighted = ((f1 * support) / support.sum()).sum().item()
    else:
        f1_weighted = 0.0

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }
