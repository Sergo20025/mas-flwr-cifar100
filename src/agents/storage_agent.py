from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class StorageAgent:
    def __init__(self, runs_dir: str = "runs", ckpt_dir: str = "checkpoints") -> None:
        self.runs_dir = Path(runs_dir)
        self.ckpt_dir = Path(ckpt_dir)

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.runs_dir / "history.json"

    def save_history(self, history: dict[str, Any]) -> None:
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def save_checkpoint(self, round_num: int, model_state_dict: dict[str, Any]) -> str:
        ckpt_path = self.ckpt_dir / f"round_{round_num}.pt"
        torch.save(model_state_dict, ckpt_path)
        return str(ckpt_path)

    def save_best_checkpoint(self, model_state_dict: dict[str, Any]) -> str:
        ckpt_path = self.ckpt_dir / "best_model.pt"
        torch.save(model_state_dict, ckpt_path)
        return str(ckpt_path)