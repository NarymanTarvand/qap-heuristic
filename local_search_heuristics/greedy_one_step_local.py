from itertools import combinations

import numpy as np

from global_calculations import (
    calculate_objective,
    read_instance_data,
    read_optimal_solution,
)

from local_utils import calculate_neighbourhood_optimal_solution

def generate_one_adjacent_swap_neighbourhood(solution_encoding: list, n: int) -> list:
    neighbourhood = []
    for idx, _ in enumerate(solution_encoding):
        neighbour = solution_encoding.copy()
        neighbour[idx], neighbour[(idx + 1) % n] = neighbour[(idx + 1) % n], neighbour[idx]
        neighbourhood.append(neighbour)
    return neighbourhood

def adjacent_swap_generator(solution_encoding: list, n: int):
    idx = 0
    neighbour = solution_encoding.copy()
    while idx + 1 <= n:
        neighbour[idx], neighbour[(idx + 1) % n] = neighbour[(idx + 1) % n], neighbour[idx]
        yield neighbour
        neighbour[(idx + 1) % n], neighbour[idx] = neighbour[idx], neighbour[(idx + 1) % n]
        i += 1

def generate_one_total_swap_neighbourhood(solution_encoding: list, n: int) -> list:
    """ generates n choose 2 neighbouring swaps """
    neighbourhood = []
    for idx1, idx2 in combinations(range(n), 2):
        neighbour = solution_encoding.copy()
        neighbour[idx1], neighbour[idx2] = neighbour[idx2], neighbour[idx1]
        neighbourhood.append(neighbour)
    return neighbourhood

def greedy_local_search(
    solution_encoding: list, flow: np.array, distance: np.array,
    neighbourhood_builder) -> list:

    current_objective = calculate_objective(solution_encoding, flow, distance)

    while True:
        neighbourhood = neighbourhood_builder(solution_encoding, n)
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
    n = len(flow)
    solution_encoding = [i for i in range(n)]

    read_optimal_solution("data/qapsoln/tai30b.sln")
    optimal_local_search_solution, objective = greedy_local_search(
        solution_encoding, flow, distance, generate_one_adjacent_swap_neighbourhood
    )
    print(
        f"Optimal solution, {optimal_local_search_solution} has objective {objective}"
    )
