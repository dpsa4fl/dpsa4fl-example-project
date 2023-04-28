from typing import List, Tuple

import flwr as fl
import dpsa4flower
from flwr.common import Metrics
from flwr.server import client_manager


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

client_manager = fl.server.SimpleClientManager()

# Start Flower server
fl.server.start_server(
    server=dpsa4flower.DPSAServer(
        62006,
        30,
        32,
        "http://127.0.0.1:9981",
        "http://127.0.0.1:9982",
        client_manager=client_manager,
        strategy=strategy),
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=3),
)

