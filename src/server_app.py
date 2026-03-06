from __future__ import annotations

<<<<<<< HEAD
=======
import os

>>>>>>> 0a736d8f481586f99da1faeedb3b3e80bfaa5e25
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.agents.aggregation_agent import AggregationAgent, AgentFedAvg
from src.agents.storage_agent import StorageAgent
<<<<<<< HEAD
from src.data import load_server_test_loader
from src.logger import get_logger
from src.model import get_device, get_model
from src.train import evaluate
from src.utils import get_parameters, set_parameters


storage = StorageAgent()
logger = get_logger("ServerApp")


def server_fn(context: Context) -> ServerAppComponents:
=======
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

>>>>>>> 0a736d8f481586f99da1faeedb3b3e80bfaa5e25
    run_config = context.run_config

    num_rounds = int(run_config["num-server-rounds"])
    fraction_fit = float(run_config["fraction-fit"])
<<<<<<< HEAD
    min_fit_clients = int(run_config["min-fit-clients"])
    min_available_clients = int(run_config["min-available-clients"])

    local_epochs = int(run_config["local-epochs"])
    learning_rate = float(run_config["learning-rate"])

    logger.info(
        f"Server initialized | rounds={num_rounds} | lr={learning_rate} | local_epochs={local_epochs}"
    )

    device = get_device()
    logger.info(f"Server evaluation device: {device}")

    model = get_model(num_classes=100).to(device)
    test_loader = load_server_test_loader(batch_size=256, num_workers=0)

    def fit_config(server_round: int):
        logger.info(f"ROUND {server_round} START")
        cfg = {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "lr": learning_rate,
        }
        logger.info(f"ROUND {server_round} -> sending FIT config to clients")
        return cfg

    storage.save_checkpoint(0, model.state_dict())
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    logger.info("Initial global model created")

    aggregation_agent = AggregationAgent(model=model, storage=storage)

    def server_evaluate(server_round: int, parameters, config):
        logger.info(f"ROUND {server_round} -> server-side evaluation START")

        # parameters is already a list of ndarrays
        set_parameters(model, parameters)

        metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=100,
        )

        logger.info(
            f"ROUND {server_round} -> server-side evaluation END | "
            f"loss={metrics['loss']:.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"f1_weighted={metrics['f1_weighted']:.4f}"
        )

        aggregation_agent.on_server_evaluate_end(
            server_round=server_round,
            metrics=metrics,
        )

        return float(metrics["loss"]), {
            "accuracy": float(metrics["accuracy"]),
            "f1_macro": float(metrics["f1_macro"]),
            "f1_weighted": float(metrics["f1_weighted"]),
        }

    strategy = AgentFedAvg(
        aggregation_agent=aggregation_agent,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=0,
        min_available_clients=min_available_clients,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=aggregation_agent.weighted_average,
        evaluate_fn=server_evaluate,
=======
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
>>>>>>> 0a736d8f481586f99da1faeedb3b3e80bfaa5e25
        initial_parameters=initial_parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)

<<<<<<< HEAD
    logger.info("Strategy initialized, simulation starting")

=======
>>>>>>> 0a736d8f481586f99da1faeedb3b3e80bfaa5e25
    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


app = ServerApp(server_fn=server_fn)