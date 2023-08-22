import numpy as np
from global_calculations import (
    calculate_objective,
    read_instance_data,
    read_optimal_solution,
)


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


def calculate_neighbourhood_optimal_solution(
    neighbourhood: list, flow: np.array, distance: np.array
) -> tuple[list, float]:
    objective_values = []
    for candidate_solution in neighbourhood:
        objective_values.append(calculate_objective(candidate_solution, flow, distance))
    optimal_neighbour = neighbourhood[np.argmin(objective_values)]
    objective_value = np.min(objective_values)

    return optimal_neighbour, objective_value


def greedy_one_step_local(
    solution_encoding: list, flow: np.array, distance: np.array
) -> list:

    current_objective = calculate_objective(solution_encoding, flow, distance)

    while True:
        neighbourhood = generate_one_swap_neighbourhood(solution_encoding)
        (
            candidate_solution,
            candidate_objective,
        ) = calculate_neighbourhood_optimal_solution(neighbourhood, flow, distance)

        if candidate_objective < current_objective:
            current_objective = candidate_objective
            solution_encoding = candidate_solution
        else:
            break
    return solution_encoding, current_objective


if __name__ == "__main__":
    instance_filepath = "data/qapdata/tai30b.dat"
    flow, distance = read_instance_data(instance_filepath)

    solution_encoding = [i for i in range(len(flow))]

    read_optimal_solution("data/qapsoln/tai30b.sln")
    optimal_local_search_solution, objective = greedy_one_step_local(
        solution_encoding, flow, distance
    )
    print(
        f"Optimal solution, {optimal_local_search_solution} has objective {objective}"
    )
