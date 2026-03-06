from __future__ import annotations

from typing import Any

from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from src.agents.storage_agent import StorageAgent
from src.utils import set_parameters


class AggregationAgent:
    """
    Agent responsible for:
    - metric aggregation
    - updating global model after aggregation
    - delegating checkpoint/history saving to StorageAgent
    """

    def __init__(self, model, storage: StorageAgent) -> None:
        self.model = model
        self.storage = storage
        self.history_data = {
            "loss_distributed": [],
            "metrics_distributed": {},
        }

    def weighted_average(self, metrics):
        """Aggregate numeric metrics weighted by number of examples."""
        if not metrics:
            return {}

        total_examples = sum(num_examples for num_examples, _ in metrics)
        aggregated = {}

        keys = set()
        for _, metric_dict in metrics:
            keys.update(metric_dict.keys())

        for key in keys:
            weighted_sum = 0.0
            found_numeric = False

            for num_examples, metric_dict in metrics:
                value = metric_dict.get(key)
                if isinstance(value, (int, float)):
                    weighted_sum += num_examples * float(value)
                    found_numeric = True

            if found_numeric and total_examples > 0:
                aggregated[key] = weighted_sum / total_examples

        return aggregated

    def handle_aggregate_fit(self, server_round: int, aggregated_parameters) -> None:
        """Update global model and save checkpoint after fit aggregation."""
        if aggregated_parameters is None:
            return

        ndarrays = parameters_to_ndarrays(aggregated_parameters)
        set_parameters(self.model, ndarrays)
        self.storage.save_checkpoint(server_round, self.model.state_dict())

    def handle_aggregate_evaluate(
        self,
        server_round: int,
        aggregated_loss: float | None,
        aggregated_metrics: dict[str, Any] | None,
    ) -> None:
        """Save aggregated evaluation history."""
        if aggregated_loss is not None:
            self.history_data["loss_distributed"].append(
                {"round": server_round, "loss": float(aggregated_loss)}
            )

        if aggregated_metrics:
            for key, value in aggregated_metrics.items():
                if isinstance(value, (int, float)):
                    self.history_data["metrics_distributed"].setdefault(key, [])
                    self.history_data["metrics_distributed"][key].append(
                        {"round": server_round, "value": float(value)}
                    )

        self.storage.save_history(self.history_data)


class AgentFedAvg(FedAvg):
    """
    FedAvg strategy wrapped with AggregationAgent.
    """

    def __init__(self, aggregation_agent: AggregationAgent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregation_agent = aggregation_agent

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        self.aggregation_agent.handle_aggregate_fit(
            server_round, aggregated_parameters
        )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        self.aggregation_agent.handle_aggregate_evaluate(
            server_round, aggregated_loss, aggregated_metrics
        )

        return aggregated_loss, aggregated_metrics