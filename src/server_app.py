from __future__ import annotations

import flwr as fl


def main() -> None:
    num_clients = 10
    num_rounds = 5

    def fit_config(server_round: int):
        return {"local_epochs": 1, "lr": 0.01}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
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