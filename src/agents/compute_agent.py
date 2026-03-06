from __future__ import annotations

<<<<<<< HEAD
from typing import Any

from src.agents.storage_agent import StorageAgent
from src.logger import get_logger
from src.model import get_device, get_model
from src.train import train_local
from src.utils import get_parameters, set_parameters


class ComputeAgent:
    def __init__(
        self,
        cid_raw: Any,
        num_clients: int = 10,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.num_clients = int(num_clients)
        self.cid = int(cid_raw)

        self.logger = get_logger(f"ComputeAgent[{self.cid}]")
        self.device = get_device()
        self.model = get_model(num_classes=100).to(self.device)

        self.logger.info(
            f"Initialized | raw_cid={cid_raw} | partition={self.cid} | device={self.device}"
        )

        self.storage = StorageAgent()
        self.data = self.storage.get_client_data(
            client_id=self.cid,
            num_clients=self.num_clients,
            batch_size=int(batch_size),
            seed=seed,
            num_workers=0,
        )

        self.logger.info(
            f"Partition attached | partition={self.cid} | train={self.data.num_train}"
        )

    def get_parameters(self):
        self.logger.info("Sending local model parameters")
        return get_parameters(self.model)

    def fit(
        self,
        parameters,
        local_epochs: int = 1,
        lr: float = 0.01,
        round_num: int = -1,
    ):
        self.logger.info(
            f"FIT START | round={round_num} | partition={self.cid}"
        )
        self.logger.info("Received global model")

        # Получаем глобальные веса от сервера
        set_parameters(self.model, parameters)

        # Локальное обучение на клиенте
        metrics = train_local(
            model=self.model,
            loader=self.data.train_loader,
            device=self.device,
            local_epochs=local_epochs,
            lr=lr,
        )

        # Логи по эпохам
        epoch_losses = metrics["epoch_losses"]
        for epoch_idx, epoch_loss in enumerate(epoch_losses, start=1):
            self.logger.info(
                f"TRAIN epoch={epoch_idx}/{local_epochs} | round={round_num} | loss={epoch_loss:.4f}"
            )

        self.logger.info(
            "FIT END | "
            f"round={round_num} | "
            f"sending weights | "
            f"samples={self.data.num_train} | "
            f"train_loss_last={metrics['train_loss_last']:.4f} | "
            f"train_loss_mean={metrics['train_loss_mean']:.4f}"
        )

        return self.get_parameters(), self.data.num_train, {
            "train_loss": float(metrics["train_loss_last"]),
        }
=======
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
>>>>>>> 0a736d8f481586f99da1faeedb3b3e80bfaa5e25
