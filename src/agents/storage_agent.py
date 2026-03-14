"""Агент хранения: данные клиентов, история и чекпоинты."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data import ClientData, load_cifar100_partitioned
from src.logger import get_logger


class StorageAgent:
    def __init__(
        self,
        runs_dir: str = "runs",
        ckpt_dir: str = "checkpoints",
    ) -> None:
        self.logger = get_logger("StorageAgent")

        self.runs_dir = Path(runs_dir)
        self.ckpt_dir = Path(ckpt_dir)

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.runs_dir / "history.json"

        self.logger.info(
            f"Initialized | runs_dir={self.runs_dir} | ckpt_dir={self.ckpt_dir}"
        )

    def get_client_data(
        self,
        client_id: int,
        num_clients: int,
        batch_size: int,
        seed: int = 42,
        num_workers: int = 0,
        partition_mode: str = "iid",
        dirichlet_alpha: float = 0.3,
    ) -> ClientData:
        # Подготавливаем train-данные для конкретного клиента.
        self.logger.info(
            "Preparing client data | "
            f"client_id={client_id} | num_clients={num_clients} | "
            f"batch_size={batch_size} | partition_mode={partition_mode} | "
            f"dirichlet_alpha={dirichlet_alpha}"
        )

        data = load_cifar100_partitioned(
            client_id=client_id,
            num_clients=num_clients,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            partition_mode=partition_mode,
            dirichlet_alpha=dirichlet_alpha,
        )

        self.logger.info(
            f"Client data ready | client_id={client_id} | num_train={data.num_train}"
        )
        return data

    def save_history(self, history: dict[str, Any]) -> None:
        # Сохраняем историю обучения в JSON.
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        self.logger.info(f"History saved | path={self.history_path}")

    def save_checkpoint(self, round_num: int, model_state_dict: dict[str, Any]) -> str:
        ckpt_path = self.ckpt_dir / f"round_{round_num}.pt"
        torch.save(model_state_dict, ckpt_path)
        self.logger.info(f"Checkpoint saved | round={round_num} | path={ckpt_path}")
        return str(ckpt_path)

    def save_best_checkpoint(self, model_state_dict: dict[str, Any]) -> str:
        ckpt_path = self.ckpt_dir / "best_model.pt"
        torch.save(model_state_dict, ckpt_path)
        self.logger.info(f"Best checkpoint updated | path={ckpt_path}")
        return str(ckpt_path)
