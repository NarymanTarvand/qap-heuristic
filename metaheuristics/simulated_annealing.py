import numpy as np
import random
from global_calculations import calculate_objective


def find_random_neighbour(current_solution):
    potential_solution = current_solution.copy()
    idx = range(len(potential_solution))
    i1, i2 = random.sample(idx, 2)
    potential_solution[i1], potential_solution[i2] = (
        potential_solution[i2],
        potential_solution[i1],
    )

    return potential_solution


def simmulated_annealing(
    current_solution: list,
    flow: np.array,
    distance: np.array,
    n_iter: int = 500,
    temp: int = 100,
    alpha: float = 0.99,
):
    current_objective = calculate_objective(current_solution, flow, distance)

    for _ in range(n_iter):
        candidate_solution = find_random_neighbour(current_solution)
        candidate_objective = calculate_objective(candidate_solution, flow, distance)

        if candidate_objective < current_objective:
            current_solution = candidate_solution
            current_objective = candidate_objective
        else:
            acceptance_probability = np.exp(
                (current_objective - candidate_objective) / temp
            )
            r = np.random.random()
            if acceptance_probability <= r:
                current_solution = candidate_solution
                current_objective = candidate_objective

        temp *= alpha
    return current_solution, current_objective
