import sys

import numpy as np

from global_calculations import calculate_objective, calculate_objective_vectorised
from local_search_heuristics.neighbours import swap

def calculate_neighbourhood_optimal_solution(
    neighbourhood: list, flow: np.array, distance: np.array, obj: int = -1
) -> tuple[list, float]:
    obj_values = [
        calculate_objective(neighbour, flow, distance) for neighbour in neighbourhood
    ]
    optimal_neighbour = neighbourhood[np.argmin(obj_values)]
    optimal_obj = np.min(obj_values)

    return optimal_neighbour, optimal_obj


def calculate_neighbourhood_first_improvement(
    neighbourhood: list, flow: np.array, distance: np.array, obj: int
) -> tuple[list, float]:

    for selected_neighbour in neighbourhood:
        curr_obj = calculate_objective(selected_neighbour, flow, distance)
        if curr_obj < obj:
            break

    return selected_neighbour, curr_obj


def compute_average_similarity(candidate: np.array, local_optima: list) -> float:
    n_optima = len(local_optima)
    if n_optima == 0:
        return 0
    similarity = sum([len(np.where(candidate == optima)[0]) for optima in local_optima])
    return (1 / n_optima) * similarity


def calculate_neighbourhood_improving_similar(
    neighbourhood: list, flow: np.array, distance, local_optima: list, current_encoding: list, current_obj: int
) -> tuple[list, float]:
    
    obj_values = [
        calculate_objective_vectorised(current_encoding, flow, distance, swap_idx[0], swap_idx[1]) for swap_idx in neighbourhood
    ]

    improving_idx = [
        i for i in range(len(neighbourhood)) if obj_values[i] < current_obj
    ]

    if not improving_idx:
        return current_encoding, current_obj
    
    neighbourhood_solutions = list()
    for idx1, idx2 in neighbourhood:
        neighbour = current_encoding.copy()
        swap(neighbour, idx1, idx2)
        neighbourhood_solutions.append(neighbour)
    
    similarity_vector = [
        compute_average_similarity(j, local_optima)
        for i, j in enumerate(neighbourhood_solutions)
        if i in improving_idx
    ]

    selected_index = improving_idx[np.argmin(similarity_vector)]

    return neighbourhood_solutions[selected_index], obj_values[selected_index]


if __name__ == "__main__":
    pass
