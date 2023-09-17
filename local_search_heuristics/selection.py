import sys

import numpy as np

from global_calculations import (
    calculate_objective, calculate_objective_incremental
)

def calculate_neighbourhood_optimal_solution(
    neighbourhood: list, flow: np.array, distance: np.array
) -> tuple[list, float]:
    obj_values = [calculate_objective(neighbour, flow, distance) for neighbour in neighbourhood]
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

def compute_average_similarity(
        candidate: np.array, local_optima: list
) -> float:
    n_optima = len(local_optima)
    similarity = sum([len(np.where(candidate==optima)[0]) for optima in local_optima])
    return (1/n_optima) * similarity

def calculate_neighbourhood_improving_similar(
        neighbourhood: list, flow: np.array, distance, local_optima: list,
        current_obj: int) -> tuple[list, float]:
    
    obj_values = [calculate_objective(neighbour, flow, distance) for neighbour in neighbourhood]
    
    improving_idx = [i for i in range(len(neighbourhood)) if obj_values[i] < current_obj]
    
    if not improving_idx:
        return neighbourhood[0], obj_values[0]
    else:
        similarity_vector = np.apply_along_axis(lambda x: compute_average_similarity(x, local_optima), 0, 
                                                [i for i, j in enumerate(neighbourhood) if j in improving_idx])
        
        selected_index = neighbourhood[np.argmin(similarity_vector)]
        
        return neighbourhood[selected_index], obj_values[selected_index]

if __name__ == "__main__":
    pass