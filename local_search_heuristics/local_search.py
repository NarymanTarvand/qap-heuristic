import random
from typing import Tuple, List, Callable

import numpy as np

from global_calculations import (
    calculate_objective,
    read_instance_data,
    read_optimal_solution,
)

from local_search_heuristics.selection import *
from local_search_heuristics.neighbours import *


def local_search(
    solution_encoding: list,
    flow: np.array,
    distance: np.array,
    neighbourhood_builder: Callable = get_total_swap_neighbourhood,
    solution_selector: Callable = calculate_neighbourhood_optimal_solution,
) -> Tuple[List[int], float]:

    current_objective = calculate_objective(solution_encoding, flow, distance)
    n = len(flow)

    while True:
        neighbourhood = neighbourhood_builder(solution_encoding, n)
        (
            candidate_solution,
            candidate_objective,
        ) = solution_selector(neighbourhood, flow, distance, current_objective)

        if candidate_objective < current_objective:
            current_objective = candidate_objective
            solution_encoding = candidate_solution
        else:
            break

    return solution_encoding, current_objective


def multistart(
    flow: np.array,
    distance: np.array,
    neighbourhood_builder: Callable,
    k: int,
    solution_selector: Callable = calculate_neighbourhood_optimal_solution,
    deterministic_start: list = [],
) -> Tuple[List[int], float]:

    # TODO: refactor for readability
    multistart_solutions = []
    multistart_costs = []
    n = len(flow)

    given_start = False
    if deterministic_start != []:
        given_start = True
        assert len(deterministic_start) == k

    for j in range(k):
        if given_start:
            solution_encoding = deterministic_start[j]
        else:
            solution_encoding = [i for i in range(n)]
            random.shuffle(solution_encoding)
        solution_encoding, solution_cost = local_search(
            solution_encoding, flow, distance, neighbourhood_builder, solution_selector
        )
        multistart_solutions.append(solution_encoding)
        multistart_costs.append(solution_cost)

    multistart_cost = np.min(multistart_costs)
    multistart_solution = multistart_costs[np.argmin(multistart_costs)]

    return multistart_solution, multistart_cost


def disimilarity_local_search(
    flow: np.array,
    distance: np.array,
    neighbourhood_builder: Callable,
    k: int,
    deterministic_start: list = [],
) -> Tuple[List[int], float]:

    n = len(flow)
    local_optima = []
    optima_costs = []

    given_start = False
    if deterministic_start != []:
        given_start = True
        assert len(deterministic_start) == k

    for j in range(k):
        # initial random feasible solution or given deterministic
        if given_start:
            solution_encoding = deterministic_start[j]
        else:
            solution_encoding = [i for i in range(n)]
            random.shuffle(solution_encoding)

        current_objective = calculate_objective(solution_encoding, flow, distance)

        while True:
            neighbourhood = neighbourhood_builder(solution_encoding, n)
            (
                candidate_solution,
                candidate_objective,
            ) = calculate_neighbourhood_improving_similar(
                neighbourhood, flow, distance, local_optima, current_objective
            )

            if candidate_objective < current_objective:
                current_objective = candidate_objective
                solution_encoding = candidate_solution
            else:
                break

        local_optima.append(solution_encoding)
        optima_costs.append(current_objective)

    multistart_disimilarity_cost = np.min(optima_costs)
    multistart_disimilarity_solution = local_optima[np.argmin(optima_costs)]

    return multistart_disimilarity_solution, multistart_disimilarity_cost


if __name__ == "__main__":
    instance_filepath = "../data/qapdata/tai30b.dat"
    flow, distance = read_instance_data(instance_filepath)
    n = len(flow)
    k = 2

    best_obj, _ = read_optimal_solution("../data/qapsoln/tai30b.sln")

    optimal_local_search_solution, objective = disimilarity_local_search(
        flow, distance, get_total_swap_neighbourhood, k
    )

    print(
        f"Optimal solution, {optimal_local_search_solution} has objective {objective}"
    )
    print(f"solution: {best_obj}")
