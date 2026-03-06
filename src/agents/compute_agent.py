from __future__ import annotations

import hashlib
from typing import Any

from src.data import load_cifar100_iid
from src.model import get_device, get_model
from src.train import evaluate, train_one_epoch
from src.utils import get_parameters, set_parameters


def normalize_cid(cid_raw: Any, num_clients: int) -> int:
    """Map any Flower/Ray client id to a stable partition id in [0, num_clients-1]."""
    s = str(cid_raw)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h, 16) % num_clients


class ComputeAgent:
    """
    Compute agent:
    - owns local model
    - owns local data partition
    - performs local training/evaluation
    """

    def __init__(self, cid_raw: Any, num_clients: int = 10, batch_size: int = 64):
        self.num_clients = int(num_clients)
        self.cid = normalize_cid(cid_raw, self.num_clients)

        self.device = get_device()
        self.model = get_model(num_classes=100).to(self.device)

        self.data = load_cifar100_iid(
            client_id=self.cid,
            num_clients=self.num_clients,
            batch_size=int(batch_size),
        )

    def get_parameters(self):
        return get_parameters(self.model)

    def fit(self, parameters, local_epochs: int = 1, lr: float = 0.01):
        set_parameters(self.model, parameters)

        for _ in range(local_epochs):
            train_one_epoch(
                self.model,
                self.data.train_loader,
                self.device,
                lr=lr,
            )

        return self.get_parameters(), self.data.num_train, {}

    def evaluate(self, parameters):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.data.test_loader, self.device)
        return float(loss), self.data.num_test, {"accuracy": float(acc)}