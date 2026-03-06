from __future__ import annotations

import os
import flwr as fl


def fit_config(server_round: int):
    return {
        "local_epochs": 1,
        "lr": 0.01,
    }


def weighted_average(metrics):
    """Aggregate client metrics weighted by number of examples."""
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)

    aggregated = {}
    metric_keys = set()
    for _, m in metrics:
        metric_keys.update(m.keys())

    for key in metric_keys:
        weighted_sum = 0.0
        has_numeric = False
        for num_examples, m in metrics:
            value = m.get(key)
            if isinstance(value, (int, float)):
                weighted_sum += num_examples * float(value)
                has_numeric = True
        if has_numeric and total_examples > 0:
            aggregated[key] = weighted_sum / total_examples

    return aggregated


def main() -> None:
    # Optional: reduce Ray/HF log noise on Windows
    os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    num_clients = 10
    num_rounds = 5

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    from src.client_app import client_fn

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.1},
    )


if __name__ == "__main__":
    main()