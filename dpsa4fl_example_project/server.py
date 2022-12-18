from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from flwr.server import client_manager

from dpsa4fl_bindings import controller_api__new_state, controller_api__create_session, controller_api__start_round, PyControllerState


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# create controller state
dpsa4fl_state = controller_api__new_state(20)

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
dpsa4fl_strategy = fl.server.strategy.DPSAStrategyWrapper(strategy, dpsa4fl_state, 62006)

client_manager = fl.server.SimpleClientManager()

# Start Flower server
fl.server.start_server(
    server=fl.server.DPSAServer(dpsa4fl_state, client_manager=client_manager, strategy=dpsa4fl_strategy),
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=dpsa4fl_strategy,
)
