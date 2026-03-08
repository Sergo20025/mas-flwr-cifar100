from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.agents.aggregation_agent import DecentralizedAggregationAgent
from src.agents.compute_agent import ComputeAgent
from src.agents.storage_agent import StorageAgent
from src.data import load_server_test_loader
from src.logger import get_logger
from src.model import get_device, get_model
from src.train import evaluate
from src.utils import get_parameters, set_parameters

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


logger = get_logger("DecentralizedSim")


@dataclass
class Node:
    node_id: int
    storage: StorageAgent
    compute: ComputeAgent
    aggregation: DecentralizedAggregationAgent
    num_train: int
    params: list[np.ndarray]


def _load_run_config(path: str = "pyproject.toml") -> dict:
    pyproject_path = Path(path)
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    return data["tool"]["flwr"]["app"]["config"]


def _ring_neighbors(node_id: int, num_nodes: int) -> list[int]:
    if num_nodes <= 1:
        return [node_id]
    if num_nodes == 2:
        return [node_id, 1 - node_id]

    left = (node_id - 1 + num_nodes) % num_nodes
    right = (node_id + 1) % num_nodes
    return [node_id, left, right]


def _clone_params(params: list[np.ndarray]) -> list[np.ndarray]:
    return [p.copy() for p in params]


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = _load_run_config()

    num_rounds = int(cfg.get("num-server-rounds", 50))
    num_nodes = int(cfg.get("num-clients", 10))
    batch_size = int(cfg.get("batch-size", 128))
    local_epochs = int(cfg.get("local-epochs", 3))
    lr = float(cfg.get("learning-rate", 0.03))
    seed = int(cfg.get("seed", 42))
    min_delta_norm = float(cfg.get("min-update-norm", 0.0))

    logger.info(
        "Decentralized FL start | "
        f"rounds={num_rounds} | nodes={num_nodes} | "
        f"epochs={local_epochs} | lr={lr} | min_delta_norm={min_delta_norm}"
    )

    device = get_device()
    logger.info(f"Server evaluator device: {device}")

    evaluator_model = get_model(num_classes=100).to(device)
    test_loader = load_server_test_loader(batch_size=256, num_workers=0)

    base_model = get_model(num_classes=100).cpu()
    initial_params = get_parameters(base_model)

    global_storage = StorageAgent(runs_dir="runs", ckpt_dir="checkpoints")
    global_storage.save_checkpoint(0, evaluator_model.cpu().state_dict())
    evaluator_model.to(device)

    nodes: list[Node] = []
    for node_id in range(num_nodes):
        node_storage = StorageAgent(
            runs_dir=f"runs/node_{node_id}",
            ckpt_dir=f"checkpoints/node_{node_id}",
        )
        compute = ComputeAgent(
            cid_raw=node_id,
            num_clients=num_nodes,
            batch_size=batch_size,
            seed=seed,
            storage=node_storage,
        )
        params = _clone_params(initial_params)
        set_parameters(compute.model, params)
        aggregation = DecentralizedAggregationAgent(
            node_id=node_id,
            min_delta_norm=min_delta_norm,
        )
        nodes.append(
            Node(
                node_id=node_id,
                storage=node_storage,
                compute=compute,
                aggregation=aggregation,
                num_train=compute.data.num_train,
                params=params,
            )
        )

    history: dict[str, list[dict[str, float]]] = {
        "decentralized_fit": [],
        "server_evaluate": [],
    }
    best_acc = 0.0

    for server_round in range(1, num_rounds + 1):
        logger.info(f"ROUND {server_round} START")

        local_updates: dict[int, tuple[list[np.ndarray], int, dict[str, float]]] = {}
        for node in nodes:
            updated_params, num_examples, fit_metrics = node.compute.fit(
                parameters=node.params,
                local_epochs=local_epochs,
                lr=lr,
                round_num=server_round,
            )
            local_updates[node.node_id] = (updated_params, int(num_examples), fit_metrics)

        new_params_per_node: dict[int, list[np.ndarray]] = {}
        accepted_count = 0
        mean_delta_norm = 0.0

        for node in nodes:
            neighbors = _ring_neighbors(node.node_id, num_nodes)
            peer_updates = [
                (local_updates[peer_id][1], local_updates[peer_id][0])
                for peer_id in neighbors
            ]

            aggregated, delta_norm, accepted = node.aggregation.aggregate_with_threshold(
                current_params=node.params,
                peer_updates=peer_updates,
            )

            mean_delta_norm += delta_norm
            if accepted:
                accepted_count += 1

            new_params_per_node[node.node_id] = aggregated

        mean_delta_norm /= float(max(len(nodes), 1))

        for node in nodes:
            node.params = _clone_params(new_params_per_node[node.node_id])
            set_parameters(node.compute.model, node.params)
            node.storage.save_checkpoint(server_round, node.compute.model.state_dict())

        avg_train_loss = float(
            sum(local_updates[n.node_id][2].get("train_loss", 0.0) for n in nodes)
            / float(max(len(nodes), 1))
        )

        global_params = DecentralizedAggregationAgent.weighted_average_parameters(
            [(node.num_train, node.params) for node in nodes]
        )
        set_parameters(evaluator_model, global_params)

        eval_metrics = evaluate(
            model=evaluator_model,
            loader=test_loader,
            device=device,
            num_classes=100,
        )

        logger.info(
            "ROUND END | "
            f"round={server_round} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"acc={eval_metrics['accuracy']:.4f} | "
            f"f1_macro={eval_metrics['f1_macro']:.4f} | "
            f"accepted_nodes={accepted_count}/{num_nodes}"
        )

        history["decentralized_fit"].append(
            {
                "round": float(server_round),
                "train_loss": avg_train_loss,
                "accepted_ratio": float(accepted_count / float(max(num_nodes, 1))),
                "mean_delta_norm": float(mean_delta_norm),
            }
        )
        history["server_evaluate"].append(
            {
                "round": float(server_round),
                "loss": float(eval_metrics["loss"]),
                "accuracy": float(eval_metrics["accuracy"]),
                "f1_macro": float(eval_metrics["f1_macro"]),
                "f1_weighted": float(eval_metrics["f1_weighted"]),
            }
        )

        set_parameters(evaluator_model, global_params)
        global_storage.save_checkpoint(server_round, copy.deepcopy(evaluator_model.cpu().state_dict()))
        evaluator_model.to(device)

        if float(eval_metrics["accuracy"]) > best_acc:
            best_acc = float(eval_metrics["accuracy"])
            global_storage.save_best_checkpoint(copy.deepcopy(evaluator_model.cpu().state_dict()))
            evaluator_model.to(device)

        _save_json(Path("runs/history.json"), history)

    logger.info("Decentralized FL finished")


if __name__ == "__main__":
    main()
