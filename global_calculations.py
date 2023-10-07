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


def calculate_objective_incremental(p, n, curr_obj, flow, distance, a, b):
    """Compute the change in the objective function after swapping facilities a and b in permutation p."""
    delta = 0

    for k in range(n):
        if k != a and k != b:
            delta += (
                flow[a][k] * (distance[p[b]][p[k]] - distance[p[a]][p[k]])
                + flow[k][a] * (distance[p[k]][p[b]] - distance[p[k]][p[a]])
                + flow[b][k] * (distance[p[a]][p[k]] - distance[p[b]][p[k]])
                + flow[k][b] * (distance[p[k]][p[a]] - distance[p[k]][p[b]])
            )

    # Account for the direct interaction between the swapped facilities
    delta += flow[a][b] * (distance[p[b]][p[a]] - distance[p[a]][p[b]]) + flow[b][a] * (
        distance[p[a]][p[b]] - distance[p[b]][p[a]]
    )

    return curr_obj + delta


def calculate_objective_vectorised(current_encoding, flow, distance, a, b):
    # swap indices
    current_encoding[a], current_encoding[b] = current_encoding[b], current_encoding[a]

    new_cost = np.sum(flow * distance[current_encoding][:, current_encoding])

    current_encoding[a], current_encoding[b] = current_encoding[b], current_encoding[a]

    # Compute and return the delta
    return new_cost
