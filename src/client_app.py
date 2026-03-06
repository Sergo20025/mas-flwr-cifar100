from __future__ import annotations

from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context

from src.agents.compute_agent import ComputeAgent


class CifarClient(NumPyClient):
    def __init__(self, agent: ComputeAgent):
        self.agent = agent

    def get_parameters(self, config):
        return self.agent.get_parameters()

    def fit(self, parameters, config):
        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.01))
        return self.agent.fit(parameters, local_epochs=local_epochs, lr=lr)

    def evaluate(self, parameters, config):
        return self.agent.evaluate(parameters)


def client_fn(context: Context):
    run_cfg = context.run_config

    num_clients = int(run_cfg["num-clients"])
    batch_size = int(run_cfg["batch-size"])

    cid_raw = None
    try:
        cid_raw = context.node_config.get("partition-id", None)
    except Exception:
        cid_raw = None

    if cid_raw is None:
        cid_raw = getattr(context, "node_id", "0")

    agent = ComputeAgent(
        cid_raw=cid_raw,
        num_clients=num_clients,
        batch_size=batch_size,
    )

    return CifarClient(agent).to_client()


app = ClientApp(client_fn=client_fn)