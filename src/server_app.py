"""Сервер Flower: оркестрация раундов и серверная оценка модели."""

from __future__ import annotations

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.agents.aggregation_agent import (
    AgentDecentralizedFlower,
    AgentFedAvg,
    AggregationAgent,
)
from src.agents.storage_agent import StorageAgent
from src.data import load_server_test_loader, load_server_train_loader
from src.logger import get_logger
from src.model import get_device, get_model
from src.train import evaluate
from src.utils import get_parameters, set_parameters


storage = StorageAgent()
logger = get_logger("ServerApp")


def _to_bool(value) -> bool:
    # Нормализуем булевы значения из run_config.
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def server_fn(context: Context) -> ServerAppComponents:
    # Точка входа серверного приложения Flower.
    run_config = context.run_config

    num_rounds = int(run_config["num-server-rounds"])
    fraction_fit = float(run_config["fraction-fit"])
    min_fit_clients = int(run_config["min-fit-clients"])
    min_available_clients = int(run_config["min-available-clients"])

    local_epochs = int(run_config["local-epochs"])
    learning_rate = float(run_config["learning-rate"])
    decentralized_mode = _to_bool(run_config.get("decentralized-mode", "true"))
    min_update_norm = float(run_config.get("min-update-norm", 0.0))
    num_clients = int(run_config["num-clients"])

    logger.info(
        f"Server initialized | rounds={num_rounds} | lr={learning_rate} | local_epochs={local_epochs}"
    )
    logger.info(
        f"Training mode | decentralized_mode={decentralized_mode} | min_update_norm={min_update_norm}"
    )

    device = get_device()
    logger.info(f"Server evaluation device: {device}")

    model = get_model(num_classes=100).to(device)
    train_loader = load_server_train_loader(batch_size=256, num_workers=0)
    test_loader = load_server_test_loader(batch_size=256, num_workers=0)

    def fit_config(server_round: int):
        # Конфиг локального обучения, который отправляется клиентам.
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
        # Оценка агрегированной модели на тестовом наборе сервера.
        logger.info(f"ROUND {server_round} -> server-side evaluation START")

        # parameters here is already a list of ndarrays
        set_parameters(model, parameters)

        train_metrics = evaluate(
            model=model,
            loader=train_loader,
            device=device,
            num_classes=100,
        )
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=100,
        )

        logger.info(
            f"ROUND {server_round} -> server-side evaluation END | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )

        aggregation_agent.on_server_evaluate_end(
            server_round=server_round,
            test_metrics=test_metrics,
            train_metrics=train_metrics,
        )

        return float(test_metrics["loss"]), {
            "accuracy": float(test_metrics["accuracy"]),
            "f1_macro": float(test_metrics["f1_macro"]),
            "f1_weighted": float(test_metrics["f1_weighted"]),
        }

    if decentralized_mode:
        strategy = AgentDecentralizedFlower(
            aggregation_agent=aggregation_agent,
            num_nodes=num_clients,
            min_update_norm=min_update_norm,
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=min_available_clients,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=aggregation_agent.weighted_average,
            evaluate_fn=server_evaluate,
            initial_parameters=initial_parameters,
        )
    else:
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
            initial_parameters=initial_parameters,
        )

    config = ServerConfig(num_rounds=num_rounds)

    logger.info("Strategy initialized, simulation starting")

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


app = ServerApp(server_fn=server_fn)
