from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


@dataclass
class ClientData:
    train_loader: DataLoader
    test_loader: DataLoader
    num_train: int
    num_test: int


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _hf_to_torch_dataset(hf_ds, transform):
    """Wrap HF dataset to return (tensor_image, label)."""
    class _Wrapper(torch.utils.data.Dataset):
        def __init__(self, ds, tfm):
            self.ds = ds
            self.tfm = tfm

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[int(idx)]
            img = item["img"]  # PIL
            y = int(item["fine_label"])
            x = self.tfm(img)
            return x, y

    return _Wrapper(hf_ds, transform)


def load_cifar100_iid(
    client_id: int,
    num_clients: int = 10,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> ClientData:
    assert 0 <= client_id < num_clients, "client_id out of range"
    _set_seed(seed)

    ds = load_dataset("uoft-cs/cifar100")

    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_ds = _hf_to_torch_dataset(ds["train"], train_tfm)
    test_ds = _hf_to_torch_dataset(ds["test"], test_tfm)

    n_train = len(train_ds)
    indices = np.arange(n_train)
    np.random.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    client_indices = splits[client_id].tolist()

    train_subset = Subset(train_ds, client_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return ClientData(
        train_loader=train_loader,
        test_loader=test_loader,
        num_train=len(train_subset),
        num_test=len(test_ds),
    )