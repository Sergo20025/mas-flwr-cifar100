"""Агент агрегации и стратегии Flower (централизованная и децентрализованная)."""

from __future__ import annotations

import numpy as np
from flwr.common import (
    FitIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.logger import get_logger
from src.utils import set_parameters


class AggregationAgent:
    def __init__(self, model, storage):
        self.logger = get_logger("AggregationAgent")
        self.model = model
        self.storage = storage

        self.history: dict[str, list[dict[str, float]]] = {
            "fit_metrics": [],
            "server_evaluate": [],
            "server_evaluate_train": [],
            "server_evaluate_test": [],
        }

        self.best_accuracy = 0.0
        self.best_f1_macro = 0.0

        self.logger.info("Aggregation agent initialized")

    def weighted_average(self, metrics):
        # Взвешенное усреднение клиентских метрик по количеству примеров.
        if not metrics:
            return {}

        total_examples = sum(num_examples for num_examples, _ in metrics)
        result: dict[str, float] = {}

        for _, metric_dict in metrics:
            for key in metric_dict:
                result.setdefault(key, 0.0)

        for key in result:
            weighted_sum = 0.0
            for num_examples, metric_dict in metrics:
                value = metric_dict.get(key, 0.0)
                if isinstance(value, (int, float)):
                    weighted_sum += num_examples * float(value)
            result[key] = weighted_sum / total_examples if total_examples > 0 else 0.0

        self.logger.info(f"Aggregated client fit metrics: {result}")
        return result

    def on_fit_end(self, server_round, aggregated_parameters, metrics):
        self.logger.info(
            f"AGGREGATION FIT | round={server_round} | updating global model"
        )

        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            set_parameters(self.model, ndarrays)
            self.storage.save_checkpoint(server_round, self.model.state_dict())

        fit_entry: dict[str, object] = {"round": float(server_round)}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                fit_entry[key] = float(value)
            elif key == "train_loss_distribution" and isinstance(value, list):
                fit_entry[key] = [float(item) for item in value]
        self.history["fit_metrics"].append(fit_entry)
        self.storage.save_history(self.history)

    def on_server_evaluate_end(
        self,
        server_round: int,
        test_metrics: dict[str, float],
        train_metrics: dict[str, float] | None = None,
    ):
        self.logger.info(
            "SERVER EVAL | "
            f"round={server_round} | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_acc={test_metrics['accuracy']:.4f} | "
            f"test_f1_macro={test_metrics['f1_macro']:.4f} | "
            f"test_f1_weighted={test_metrics['f1_weighted']:.4f}"
        )

        self.history["server_evaluate"].append(
            {
                "round": float(server_round),
                **{k: float(v) for k, v in test_metrics.items()},
            }
        )
        self.history["server_evaluate_test"].append(
            {
                "round": float(server_round),
                **{k: float(v) for k, v in test_metrics.items()},
            }
        )
        if train_metrics is not None:
            self.history["server_evaluate_train"].append(
                {
                    "round": float(server_round),
                    **{k: float(v) for k, v in train_metrics.items()},
                }
            )
        self.storage.save_history(self.history)

        if test_metrics["accuracy"] > self.best_accuracy:
            self.best_accuracy = test_metrics["accuracy"]
            self.logger.info(
                f"NEW BEST ACC MODEL | round={server_round} | acc={test_metrics['accuracy']:.4f}"
            )
            self.storage.save_best_checkpoint(self.model.state_dict())

        if test_metrics["f1_macro"] > self.best_f1_macro:
            self.best_f1_macro = test_metrics["f1_macro"]
            self.logger.info(
                f"NEW BEST F1 MODEL | round={server_round} | f1_macro={test_metrics['f1_macro']:.4f}"
            )

    def get_history(self):
        return self.history


class AgentFedAvg(FedAvg):
    def __init__(self, aggregation_agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregation_agent = aggregation_agent
        self.logger = get_logger("AgentFedAvg")

    def aggregate_fit(self, server_round, results, failures):
        self.logger.info(
            f"AGGREGATE FIT | round={server_round} | clients={len(results)} | failures={len(failures)}"
        )
        train_loss_distribution = [
            float(fit_res.metrics["train_loss"])
            for _, fit_res in results
            if "train_loss" in fit_res.metrics
        ]

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )
        aggregated_metrics = aggregated_metrics or {}
        if train_loss_distribution:
            aggregated_metrics["train_loss_distribution"] = train_loss_distribution

        self.logger.info(f"GLOBAL MODEL UPDATED | round={server_round}")

        self.aggregation_agent.on_fit_end(
            server_round,
            aggregated_parameters,
            aggregated_metrics,
        )

        return aggregated_parameters, aggregated_metrics


class DecentralizedAggregationAgent:
    def __init__(self, node_id: int, min_delta_norm: float = 0.0):
        self.node_id = int(node_id)
        self.min_delta_norm = float(min_delta_norm)
        self.logger = get_logger(f"DecentralizedAggregationAgent[{self.node_id}]")

    @staticmethod
    def weighted_average_parameters(
        updates: list[tuple[int, list[np.ndarray]]],
    ) -> list[np.ndarray]:
        if not updates:
            raise ValueError("updates cannot be empty")

        total_examples = sum(num_examples for num_examples, _ in updates)
        if total_examples <= 0:
            raise ValueError("total_examples must be positive")

        num_tensors = len(updates[0][1])
        averaged: list[np.ndarray] = []
        for tensor_idx in range(num_tensors):
            weighted = None
            for num_examples, params in updates:
                part = params[tensor_idx] * float(num_examples)
                weighted = part if weighted is None else (weighted + part)
            averaged.append(weighted / float(total_examples))
        return averaged

    @staticmethod
    def delta_l2_norm(old_params: list[np.ndarray], new_params: list[np.ndarray]) -> float:
        sq_sum = 0.0
        for old, new in zip(old_params, new_params):
            diff = new.astype(np.float64) - old.astype(np.float64)
            sq_sum += float(np.sum(diff * diff))
        return float(np.sqrt(sq_sum))

    def aggregate_with_threshold(
        self,
        current_params: list[np.ndarray],
        peer_updates: list[tuple[int, list[np.ndarray]]],
    ) -> tuple[list[np.ndarray], float, bool]:
        # Принимаем новое состояние только при изменении выше порога.
        candidate = self.weighted_average_parameters(peer_updates)
        delta_norm = self.delta_l2_norm(current_params, candidate)
        accepted = delta_norm >= self.min_delta_norm

        if not accepted:
            self.logger.info(
                f"Skip aggregate | node={self.node_id} | delta_norm={delta_norm:.8f} < threshold={self.min_delta_norm:.8f}"
            )
            return current_params, delta_norm, False

        self.logger.info(
            f"Aggregate accepted | node={self.node_id} | delta_norm={delta_norm:.8f}"
        )
        return candidate, delta_norm, True


class AgentDecentralizedFlower(FedAvg):
    def __init__(
        self,
        aggregation_agent: AggregationAgent,
        num_nodes: int,
        min_update_norm: float = 0.0,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            *args,
            **kwargs,
        )
        self.logger = get_logger("AgentDecentralizedFlower")
        self.aggregation_agent = aggregation_agent
        self.num_nodes = int(num_nodes)
        self.min_update_norm = float(min_update_norm)
        self.node_params: dict[int, NDArrays] = {}
        self.node_examples: dict[int, int] = {}
        self.global_params: NDArrays | None = None
        self.proxy_to_node: dict[str, int] = {}

    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Parameters | None:
        initial = super().initialize_parameters(client_manager)
        if initial is None:
            return None

        init_nd = parameters_to_ndarrays(initial)
        self.node_params = {
            node_id: [arr.copy() for arr in init_nd]
            for node_id in range(self.num_nodes)
        }
        self.global_params = [arr.copy() for arr in init_nd]
        return initial

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        # Отправляем каждому клиенту параметры его логического узла.
        if server_round == 1 and not self.node_params:
            init_nd = parameters_to_ndarrays(parameters)
            self.node_params = {
                node_id: [arr.copy() for arr in init_nd]
                for node_id in range(self.num_nodes)
            }
            self.global_params = [arr.copy() for arr in init_nd]

        config: dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        fit_cfg: list[tuple[ClientProxy, FitIns]] = []
        for client in clients:
            mapped_node = self.proxy_to_node.get(client.cid)
            node_nd = (
                self.node_params.get(mapped_node, self.global_params)
                if mapped_node is not None
                else self.global_params
            )
            if node_nd is None:
                node_nd = parameters_to_ndarrays(parameters)

            node_params = ndarrays_to_parameters(node_nd)
            fit_cfg.append((client, FitIns(node_params, dict(config))))

        return fit_cfg

    def _ring_neighbors(self, node_id: int) -> list[int]:
        # Кольцевая топология: текущий узел плюс левый и правый соседи.
        if self.num_nodes <= 1:
            return [node_id]
        if self.num_nodes == 2:
            return [node_id, 1 - node_id]
        return [
            node_id,
            (node_id - 1 + self.num_nodes) % self.num_nodes,
            (node_id + 1) % self.num_nodes,
        ]

    def _weighted_average(self, updates: list[tuple[int, NDArrays]]) -> NDArrays:
        total_examples = sum(num_examples for num_examples, _ in updates)
        if total_examples <= 0:
            return [arr.copy() for arr in updates[0][1]]

        num_tensors = len(updates[0][1])
        averaged: NDArrays = []
        for tensor_idx in range(num_tensors):
            weighted = None
            for num_examples, params in updates:
                part = params[tensor_idx] * float(num_examples)
                weighted = part if weighted is None else (weighted + part)
            averaged.append(weighted / float(total_examples))
        return averaged

    def _delta_norm(self, old_params: NDArrays, new_params: NDArrays) -> float:
        sq_sum = 0.0
        for old, new in zip(old_params, new_params):
            diff = new.astype(np.float64) - old.astype(np.float64)
            sq_sum += float(np.sum(diff * diff))
        return float(np.sqrt(sq_sum))

    def aggregate_fit(self, server_round, results, failures):
        # Выполняем децентрализованную агрегацию обновлений по ring-соседям.
        self.logger.info(
            f"DECENTRALIZED AGGREGATE FIT | round={server_round} | clients={len(results)} | failures={len(failures)}"
        )
        if not results:
            return None, {}

        client_updates: dict[int, tuple[NDArrays, int]] = {}
        fit_metrics: list[tuple[int, dict[str, Scalar]]] = []
        train_loss_distribution: list[float] = []

        for client_proxy, fit_res in results:
            cid = int(fit_res.metrics.get("cid", 0))
            self.proxy_to_node[client_proxy.cid] = cid

            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = int(fit_res.num_examples)
            client_updates[cid] = (ndarrays, num_examples)
            self.node_examples[cid] = num_examples
            cleaned_metrics = {
                k: v for k, v in fit_res.metrics.items() if k != "cid"
            }
            if "train_loss" in cleaned_metrics:
                train_loss_distribution.append(float(cleaned_metrics["train_loss"]))
            fit_metrics.append((num_examples, cleaned_metrics))

        new_node_params: dict[int, NDArrays] = {}
        accepted_nodes = 0

        for node_id in range(self.num_nodes):
            old_node = self.node_params.get(node_id, self.global_params)
            if old_node is None:
                continue

            peers: list[tuple[int, NDArrays]] = []
            for peer_id in self._ring_neighbors(node_id):
                if peer_id in client_updates:
                    peer_params, peer_examples = client_updates[peer_id]
                    peers.append((peer_examples, peer_params))
                else:
                    cached = self.node_params.get(peer_id, old_node)
                    cached_examples = self.node_examples.get(peer_id, 1)
                    peers.append((cached_examples, cached))

            candidate = self._weighted_average(peers)
            delta_norm = self._delta_norm(old_node, candidate)
            if delta_norm >= self.min_update_norm:
                new_node_params[node_id] = candidate
                accepted_nodes += 1
            else:
                new_node_params[node_id] = [arr.copy() for arr in old_node]

        if new_node_params:
            self.node_params = new_node_params

        global_updates: list[tuple[int, NDArrays]] = []
        for node_id, node_p in self.node_params.items():
            global_updates.append((self.node_examples.get(node_id, 1), node_p))
        self.global_params = self._weighted_average(global_updates)

        aggregated_metrics: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        if train_loss_distribution:
            aggregated_metrics["train_loss_distribution"] = train_loss_distribution

        aggregated_metrics["accepted_nodes_ratio"] = float(
            accepted_nodes / float(max(self.num_nodes, 1))
        )

        global_parameters = ndarrays_to_parameters(self.global_params)
        self.aggregation_agent.on_fit_end(
            server_round,
            global_parameters,
            aggregated_metrics,
        )
        return global_parameters, aggregated_metrics
