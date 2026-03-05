from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

import flwr as fl
from flwr.common import Context

from src.data import load_cifar100_iid
from src.model import get_model, get_device
from src.train import train_one_epoch, evaluate
from src.utils import get_parameters, set_parameters


def _normalize_cid(cid_raw: Any, num_clients: int) -> int:
    """
    Flower/Ray can pass arbitrary client identifiers (not necessarily 0..N-1).
    We map any cid to a stable integer in [0, num_clients-1].
    """
    s = str(cid_raw)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h, 16) % num_clients


class CifarClient(fl.client.NumPyClient):
    def __init__(self, cid_raw: Any, num_clients: int = 10, batch_size: int = 64):
        self.num_clients = int(num_clients)
        self.cid = _normalize_cid(cid_raw, self.num_clients)

        self.device = get_device()
        self.model = get_model(num_classes=100).to(self.device)

        # IMPORTANT: pass normalized cid
        self.data = load_cifar100_iid(
            client_id=self.cid,
            num_clients=self.num_clients,
            batch_size=int(batch_size),
        )

    def get_parameters(self, config: Dict[str, Any]):
        return get_parameters(self.model)

    def fit(self, parameters, config: Dict[str, Any]):
        set_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.01))

        for _ in range(local_epochs):
            train_one_epoch(self.model, self.data.train_loader, self.device, lr=lr)

        return get_parameters(self.model), self.data.num_train, {}

    def evaluate(self, parameters, config: Dict[str, Any]):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.data.test_loader, self.device)
        return float(loss), self.data.num_test, {"accuracy": float(acc)}


# New Flower API (preferred): client_fn(Context) -> Client
def client_fn(context: Context) -> fl.client.Client:
    """
    Factory for Flower simulation.
    Reads optional run_config from context for num_clients/batch_size.
    """
    run_cfg = context.run_config if hasattr(context, "run_config") else {}

    num_clients = int(run_cfg.get("num_clients", 10))
    batch_size = int(run_cfg.get("batch_size", 64))

    # context.node_config["partition-id"] is commonly available in newer Flower,
    # but we fall back to something stable if not present.
    cid_raw = None
    try:
        cid_raw = context.node_config.get("partition-id", None)
    except Exception:
        cid_raw = None

    if cid_raw is None:
        # fallback: use something stable-ish from context
        cid_raw = getattr(context, "node_id", "0")

    return CifarClient(cid_raw=cid_raw, num_clients=num_clients, batch_size=batch_size).to_client()