from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["sparse_categorical_accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"sparse_categorical_accuracy": sum(accuracies) / sum(examples)}


# Define strategy
# Pass parameters to the Strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, #select 100% of available clients for fitting
    fraction_evaluate=1.0, #select 100% of available clients for evaluation
    min_fit_clients=4,
    min_evaluate_clients=4,
    min_available_clients=4,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy
)

# Start simulation
# fl.simulation.start_simulation(
#    client_fn=my_client,
#    num_clients=NUM_CLIENTS,
#    config=fl.server.ServerConfig(num_rounds=3),  # Just three rounds
#    strategy=strategy,
#    client_resources=client_resources,
# )
