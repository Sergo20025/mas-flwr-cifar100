from __future__ import annotations

from typing import Any

import torch

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
        storage: StorageAgent | None = None,
    ):
        self.num_clients = int(num_clients)
        self.cid = int(cid_raw)

        self.logger = get_logger(f"ComputeAgent[{self.cid}]")
        self.device = get_device()
        # Keep model on CPU and move to GPU only during local training.
        # This enables sequential training of many agents on one GPU.
        self.model = get_model(num_classes=100).cpu()

        self.logger.info(
            f"Initialized | raw_cid={cid_raw} | partition={self.cid} | device={self.device}"
        )

        self.storage = storage or StorageAgent()
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

        set_parameters(self.model, parameters)
        self.model.to(self.device)

        metrics = train_local(
            model=self.model,
            loader=self.data.train_loader,
            device=self.device,
            local_epochs=local_epochs,
            lr=lr,
        )
        self.model.cpu()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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
            "cid": int(self.cid),
        }
