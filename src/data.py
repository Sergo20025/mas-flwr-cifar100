from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


@dataclass
class ClientData:
    train_loader: DataLoader
    num_train: int


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform, label_key: str = "fine_label") -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        item = self.hf_dataset[int(idx)]
        img = item["img"]  # PIL image
        y = int(item[self.label_key])
        x = self.transform(img)
        return x, y


def _load_dataset():
    cache_dir = Path("data") / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return load_dataset("uoft-cs/cifar100", cache_dir=str(cache_dir))


def _get_transforms():
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, test_transform


def _partition_iid_indices(
    num_samples: int,
    num_clients: int,
    seed: int,
) -> list[list[int]]:
    all_indices = np.arange(num_samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_indices)
    return [split.tolist() for split in np.array_split(all_indices, num_clients)]


def _partition_dirichlet_indices(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 10,
    max_retries: int = 20,
) -> list[list[int]]:
    if alpha <= 0:
        raise ValueError(f"dirichlet alpha must be > 0, got {alpha}")

    rng = np.random.RandomState(seed)
    num_classes = int(labels.max()) + 1
    indices = np.arange(labels.shape[0])

    for _ in range(max_retries):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):
            class_indices = indices[labels == class_id]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            class_splits = np.split(class_indices, split_points)

            for client_id, class_split in enumerate(class_splits):
                if class_split.size > 0:
                    client_indices[client_id].extend(class_split.tolist())

        min_client_size = min(len(x) for x in client_indices)
        if min_client_size >= min_size:
            return client_indices

    return client_indices


def load_cifar100_iid(
    client_id: int,
    num_clients: int,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> ClientData:
    return load_cifar100_partitioned(
        client_id=client_id,
        num_clients=num_clients,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        partition_mode="iid",
        dirichlet_alpha=0.5,
    )


def load_cifar100_partitioned(
    client_id: int,
    num_clients: int,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    partition_mode: str = "iid",
    dirichlet_alpha: float = 0.3,
) -> ClientData:
    dataset = _load_dataset()
    train_hf = dataset["train"]

    if client_id < 0 or client_id >= num_clients:
        raise ValueError(
            f"client_id must be in [0, {num_clients - 1}], got {client_id}"
        )

    train_transform, _ = _get_transforms()

    mode = partition_mode.strip().lower()
    if mode == "iid":
        split_indices = _partition_iid_indices(
            num_samples=len(train_hf),
            num_clients=num_clients,
            seed=seed,
        )
    elif mode in {"dirichlet", "non-iid", "noniid"}:
        labels = np.asarray(train_hf["fine_label"], dtype=np.int64)
        split_indices = _partition_dirichlet_indices(
            labels=labels,
            num_clients=num_clients,
            alpha=float(dirichlet_alpha),
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown partition_mode '{partition_mode}'. Use 'iid' or 'dirichlet'."
        )

    client_indices = split_indices[client_id]

    train_dataset = HFDatasetWrapper(train_hf, train_transform)
    train_subset = Subset(train_dataset, client_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return ClientData(
        train_loader=train_loader,
        num_train=len(train_subset),
    )


def load_server_test_loader(
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    dataset = _load_dataset()
    test_hf = dataset["test"]

    _, test_transform = _get_transforms()
    test_dataset = HFDatasetWrapper(test_hf, test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader
