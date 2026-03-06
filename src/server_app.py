from __future__ import annotations

import os

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.agents.aggregation_agent import AggregationAgent, AgentFedAvg
from src.agents.storage_agent import StorageAgent
from src.model import get_model
from src.utils import get_parameters


storage = StorageAgent()


def fit_config(server_round: int):
    return {
        "local_epochs": 1,
        "lr": 0.01,
    }


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

    # Initial global model
    model = get_model(num_classes=100)

    # Save round 0 checkpoint
    storage.save_checkpoint(0, model.state_dict())

    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    # Create aggregation agent
    aggregation_agent = AggregationAgent(model=model, storage=storage)

    strategy = AgentFedAvg(
        aggregation_agent=aggregation_agent,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=aggregation_agent.weighted_average,
        evaluate_metrics_aggregation_fn=aggregation_agent.weighted_average,
        initial_parameters=initial_parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


app = ServerApp(server_fn=server_fn)