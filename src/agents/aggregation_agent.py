from __future__ import annotations

from flwr.common import parameters_to_ndarrays
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
        }

        self.best_accuracy = 0.0
        self.best_f1_macro = 0.0

        self.logger.info("Aggregation agent initialized")

    def weighted_average(self, metrics):
        if not metrics:
            return {}

        total_examples = sum(num_examples for num_examples, _ in metrics)
        result = {}

        for _, metric_dict in metrics:
            for key in metric_dict:
                result.setdefault(key, 0.0)

        for key in result:
            weighted_sum = 0.0
            for num_examples, metric_dict in metrics:
                weighted_sum += num_examples * float(metric_dict.get(key, 0.0))
            result[key] = weighted_sum / total_examples

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

        self.history["fit_metrics"].append(
            {
                "round": float(server_round),
                **{k: float(v) for k, v in metrics.items()},
            }
        )
        self.storage.save_history(self.history)

    def on_server_evaluate_end(self, server_round: int, metrics: dict[str, float]):
        self.logger.info(
            "SERVER EVAL | "
            f"round={server_round} | "
            f"loss={metrics['loss']:.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"f1_weighted={metrics['f1_weighted']:.4f}"
        )

        self.history["server_evaluate"].append(
            {
                "round": float(server_round),
                **{k: float(v) for k, v in metrics.items()},
            }
        )
        self.storage.save_history(self.history)

        if metrics["accuracy"] > self.best_accuracy:
            self.best_accuracy = metrics["accuracy"]
            self.logger.info(
                f"NEW BEST ACC MODEL | round={server_round} | acc={metrics['accuracy']:.4f}"
            )
            self.storage.save_best_checkpoint(self.model.state_dict())

        if metrics["f1_macro"] > self.best_f1_macro:
            self.best_f1_macro = metrics["f1_macro"]
            self.logger.info(
                f"NEW BEST F1 MODEL | round={server_round} | f1_macro={metrics['f1_macro']:.4f}"
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

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        self.logger.info(f"GLOBAL MODEL UPDATED | round={server_round}")

        self.aggregation_agent.on_fit_end(
            server_round,
            aggregated_parameters,
            aggregated_metrics or {},
        )

        return aggregated_parameters, aggregated_metrics