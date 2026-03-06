from __future__ import annotations

import hashlib
from typing import Any

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord
from flwr.clientapp import ClientApp
from flwr.client import NumPyClient

from src.data import load_cifar100_iid
from src.model import get_device, get_model
from src.train import evaluate, train_one_epoch
from src.utils import get_parameters, set_parameters


def _normalize_cid(cid_raw: Any, num_clients: int) -> int:
    s = str(cid_raw)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h, 16) % num_clients


class CifarClient(NumPyClient):
    def __init__(self, cid_raw: Any, num_clients: int = 10, batch_size: int = 64):
        self.num_clients = int(num_clients)
        self.cid = _normalize_cid(cid_raw, self.num_clients)

        self.device = get_device()
        self.model = get_model(num_classes=100).to(self.device)

        self.data = load_cifar100_iid(
            client_id=self.cid,
            num_clients=self.num_clients,
            batch_size=int(batch_size),
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.01))

        for _ in range(local_epochs):
            train_one_epoch(self.model, self.data.train_loader, self.device, lr=lr)

        return get_parameters(self.model), self.data.num_train, {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.data.test_loader, self.device)
        return float(loss), self.data.num_test, {"accuracy": float(acc)}


def client_fn(context: Context):
    run_cfg = context.run_config

    num_clients = int(run_cfg["num-clients"])
    batch_size = int(run_cfg["batch-size"])

    cid_raw = None
    try:
        cid_raw = context.node_config.get("partition-id", None)
    except Exception:
        cid_raw = None

    if cid_raw is None:
        cid_raw = getattr(context, "node_id", "0")

    return CifarClient(
        cid_raw=cid_raw,
        num_clients=num_clients,
        batch_size=batch_size,
    ).to_client()


app = ClientApp(client_fn=client_fn)