from __future__ import annotations

<<<<<<< HEAD
from typing import Dict, List

=======
from typing import Tuple
>>>>>>> 0a736d8f481586f99da1faeedb3b3e80bfaa5e25
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


<<<<<<< HEAD
def train_local(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    local_epochs: int = 5,
    lr: float = 0.03,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    label_smoothing: float = 0.1,
) -> Dict[str, float | List[float]]:
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

=======
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> float:
    model.train()

    criterion = nn.CrossEntropyLoss()
>>>>>>> 0a736d8f481586f99da1faeedb3b3e80bfaa5e25
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
<<<<<<< HEAD
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
=======
    )

    # scheduler на одну локальную эпоху не особо нужен,
    # но он пригодится когда local_epochs > 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(len(loader), 1)
    )

    total_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    return total_loss / max(total, 1)
>>>>>>> 0a736d8f481586f99da1faeedb3b3e80bfaa5e25


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
<<<<<<< HEAD
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
=======
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)
>>>>>>> 0a736d8f481586f99da1faeedb3b3e80bfaa5e25
