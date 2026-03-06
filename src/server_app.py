from __future__ import annotations

import os

import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from src.model import get_model
from src.utils import get_parameters


def fit_config(server_round: int):
    return {
        "local_epochs": 1,
        "lr": 0.01,
    }


def weighted_average(metrics):
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


def server_fn(context: Context) -> ServerAppComponents:
    # Reduce log noise on Windows
    os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    run_config = context.run_config

    num_rounds = int(run_config["num-server-rounds"])
    fraction_fit = float(run_config["fraction-fit"])
    fraction_evaluate = float(run_config["fraction-evaluate"])
    min_fit_clients = int(run_config["min-fit-clients"])
    min_evaluate_clients = int(run_config["min-evaluate-clients"])
    min_available_clients = int(run_config["min-available-clients"])

    # Initial model parameters
    model = get_model(num_classes=100)
    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


app = ServerApp(server_fn=server_fn)