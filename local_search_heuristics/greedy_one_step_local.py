import numpy as np
from objective_functions import calculate_objective_vector_encoding


def generate_one_swap_neighbourhood(solution_encoding: list) -> list:
    neighbourhood = []
    for idx, _ in enumerate(solution_encoding):
        neighbour = solution_encoding.copy()
        if idx + 1 == len(solution_encoding):
            neighbour[idx], neighbour[0] = neighbour[0], neighbour[idx]
        else:
            neighbour[idx], neighbour[idx + 1] = neighbour[idx + 1], neighbour[idx]
        neighbourhood.append(neighbour)
    return neighbourhood


def calculate_neighbourhood_optimal_solution(neighbourhood):
    objective_values = []
    for candidate_solution in neighbourhood:
        objective_values.append(calculate_objective_vector_encoding(candidate_solution))
    optimal_neighbour = neighbourhood[np.argmin(objective_values)]
    objective_value = np.min(objective_values)

    return optimal_neighbour, objective_value


def greedy_one_step_local(solution_encoding: list) -> list:

    current_objective = calculate_objective_vector_encoding(solution_encoding)

    while True:
        neighbourhood = generate_one_swap_neighbourhood(solution_encoding)
        (
            candidate_solution,
            candidate_objective,
        ) = calculate_neighbourhood_optimal_solution(neighbourhood)

        if candidate_objective < current_objective:
            current_objective = candidate_objective
            solution_encoding = candidate_solution
        else:
            break
    return solution_encoding, current_objective


if __name__ == "__main__":
    solution_encoding = [1, 2, 3, 4, 5]

    optimal_local_search_solution, objective = greedy_one_step_local(solution_encoding)
    print(
        f"Optimal solution, {optimal_local_search_solution} has objective {objective}"
    )
