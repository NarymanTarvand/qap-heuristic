import numpy as np
from decimal import Decimal

def read_instance_data(instance_fp: str) -> tuple[np.array, np.array]:
    with open(instance_fp) as f:
        file_it = iter([int(elem) for elem in f.read().split()])

    # Number of points
    n = next(file_it)

    flow = np.array([[next(file_it) for j in range(n)] for i in range(n)], dtype=np.float64)
    distance = np.array([[next(file_it) for j in range(n)] for i in range(n)], dtype=np.float64)

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


def narys_test(current_encoding, current_objective, flow, distance, swap_idx_1, swap_idx_2):
    cost = current_objective.copy()
    pre_swap_cost_component = 0
    post_swap_cost_component = 0
    for object_i, location_i in enumerate(current_encoding):
        pre_swap_cost_component += flow[object_i, swap_idx_1] * distance[location_i, current_encoding[swap_idx_1]]
        pre_swap_cost_component += flow[object_i, swap_idx_2] * distance[location_i, current_encoding[swap_idx_2]]

        post_swap_cost_component += flow[object_i, swap_idx_1] * distance[location_i, current_encoding[swap_idx_2]]
        post_swap_cost_component += flow[object_i, swap_idx_2] * distance[location_i, current_encoding[swap_idx_1]]


    return cost + post_swap_cost_component - pre_swapz_cost_component



def calculate_objective_incremental(current_encoding, current_objective, flow, distance, a, b):
    """Compute the change in the objective function after swapping facilities a and b in permutation p."""
    n = len(current_encoding)
    delta = 0

    for i in range(n):
        for j in range(n):
            if i != a and i != b and j != a and j != b:
                # No change for these pairs
                continue
            
            # Compute the original and new costs for pairs involving a or b
            original_cost = flow[i][j] * distance[current_encoding[i]][current_encoding[j]]
            if i == a:
                new_i = b
            elif i == b:
                new_i = a
            else:
                new_i = i
            
            if j == a:
                new_j = b
            elif j == b:
                new_j = a
            else:
                new_j = j
            
            new_cost = flow[i][j] * distance[new_i][new_j]
            delta += new_cost - original_cost
            # print(delta)
            
    return current_objective + delta
