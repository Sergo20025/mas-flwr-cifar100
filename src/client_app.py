from __future__ import annotations

from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context

from src.agents.compute_agent import ComputeAgent
from src.logger import get_logger


class CifarClient(NumPyClient):
    def __init__(self, agent: ComputeAgent):
        self.agent = agent
        self.logger = get_logger(f"CifarClient[{self.agent.cid}]")

    def get_parameters(self, config):
        self.logger.info("Flower requested model parameters")
        return self.agent.get_parameters()

    def fit(self, parameters, config):
        round_num = int(config.get("server_round", -1))
        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.01))

        self.logger.info(
            f"FIT request received | round={round_num} | epochs={local_epochs} | lr={lr}"
        )

        return self.agent.fit(
            parameters,
            local_epochs=local_epochs,
            lr=lr,
            round_num=round_num,
        )

    # Клиентская evaluate отключена, сервер оценивает сам глобальную модель.


def client_fn(context: Context):
    logger = get_logger("client_fn")

    run_cfg = context.run_config
    num_clients = int(run_cfg["num-clients"])
    batch_size = int(run_cfg["batch-size"])
    seed = int(run_cfg.get("seed", 42))

    partition_id = None
    try:
        partition_id = context.node_config.get("partition-id", None)
    except Exception:
        partition_id = None

    if partition_id is None:
        partition_id = getattr(context, "node_id", 0)

    partition_id = int(partition_id)

    logger.info(
        f"Starting client | raw_partition_id={partition_id} | num_clients={num_clients} | batch_size={batch_size}"
    )

    agent = ComputeAgent(
        cid_raw=partition_id,
        num_clients=num_clients,
        batch_size=batch_size,
        seed=seed,
    )

    logger.info(f"Client initialized | partition={agent.cid}")

    return CifarClient(agent).to_client()


app = ClientApp(client_fn=client_fn)