import numpy as np
from decimal import Decimal


def read_instance_data(instance_fp: str) -> tuple[np.array, np.array]:
    with open(instance_fp) as f:
        file_it = iter([int(elem) for elem in f.read().split()])

    # Number of points
    n = next(file_it)

    flow = np.array([[next(file_it) for j in range(n)] for i in range(n)])
    distance = np.array([[next(file_it) for j in range(n)] for i in range(n)])

    return flow, distance


def read_optimal_solution(solution_fp: str) -> tuple[int, np.array]:
    with open(solution_fp) as f:
        file_it = iter([int(elem) for elem in f.read().split()])

    n = next(file_it)

    optimal_objective = next(file_it)

    optimal_encoding = [next(file_it) for _ in range(n)]
    return optimal_objective, optimal_encoding


def calculate_objective(
    solution_encoding: np.array, flow: np.array, distance: np.array
) -> int:
    cost = 0

    for object_i, location_i in enumerate(solution_encoding):
        for object_j, location_j in enumerate(solution_encoding):
            cost += flow[object_i, object_j] * distance[location_i, location_j]

    return cost


def calculate_objective_vectorised(current_encoding, flow, distance, a, b):
    # swap indices
    current_encoding[a], current_encoding[b] = current_encoding[b], current_encoding[a]

    new_cost = np.sum(flow * distance[current_encoding][:, current_encoding])

    current_encoding[a], current_encoding[b] = current_encoding[b], current_encoding[a]

    # Compute and return the delta
    return new_cost
