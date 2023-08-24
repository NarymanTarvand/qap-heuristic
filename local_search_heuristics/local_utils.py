import sys

import numpy as np

from global_calculations import calculate_objective

def calculate_neighbourhood_optimal_solution(
    neighbourhood: list, flow: np.array, distance: np.array
) -> tuple[list, float]:
    objective_values = []
    for candidate_solution in neighbourhood:
        objective_values.append(calculate_objective(candidate_solution, flow, distance))
    optimal_neighbour = neighbourhood[np.argmin(objective_values)]
    objective_value = np.min(objective_values)

    return optimal_neighbour, objective_value

def calculate_neighbourhood_first_improvement(neighbourhood, flow: np.array, 
                                              distance: np.array
                                              ) -> tuple[list, float]:
    objective_values = []
    optimal_neighbour = neighbourhood[np.argmin(objective_values)]
    objective_value = np.min(objective_values)

    return optimal_neighbour, objective_value

if __name__ == "__main__":
    pass