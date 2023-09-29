import random
from typing import Tuple, List, Callable
import numpy as np
import time

from global_calculations import (
    calculate_objective,
    calculate_objective_incremental,
    calculate_objective_incremental_vectorised,
    read_instance_data,
    read_optimal_solution,
)

from local_search_heuristics.selection import *
from local_search_heuristics.neighbours import *


# def slow_local_search(
#     solution_encoding: list,
#     flow: np.array,
#     distance: np.array,
#     neighbourhood_builder: Callable = get_total_swap_neighbourhood,
#     solution_selector: Callable = calculate_neighbourhood_optimal_solution,
# ) -> Tuple[List[int], float]:

#     current_objective = calculate_objective(solution_encoding, flow, distance)
#     n = len(flow)
#     while True:
#         neighbourhood = neighbourhood_builder(solution_encoding, n)
#         (
#             candidate_solution,
#             candidate_objective,
#         ) = solution_selector(neighbourhood, flow, distance, current_objective)

#         if candidate_objective < current_objective:
#             current_objective = candidate_objective
#             solution_encoding = candidate_solution
#         else:
#             break

#     return solution_encoding, current_objective


def local_search(
    solution_encoding: list,
    flow: np.array,
    distance: np.array,
    neighbourhood_builder: str = "total_swap",
    solution_selector: str = "best_improvement",
) -> Tuple[List[int], float]:

    current_objective = calculate_objective(solution_encoding, flow, distance)
    current_encoding = solution_encoding
    n = len(flow)

    if neighbourhood_builder == "total_swap":
        neighbourhood = list(combinations(range(n), 2))
    elif neighbourhood_builder == "adjacent_swap":
        neighbourhood = [(i, (i + 1) % (n)) for i in range(n)]
    else:
        raise Exception(
            "local_search neighbourood_builder must be one of 'total_swap', 'adjacent_swap'"
        )

    while True:
        objectives = [
            calculate_objective_incremental_vectorised(
                current_encoding,
                flow,
                distance,
                swap_idx[0],
                swap_idx[1],
            )
            for swap_idx in neighbourhood
        ]

        if solution_selector == "best_improvement":
            candidate_idx = np.argmin(objectives)
        elif solution_selector == "first_improvement":
            candidate_idx = np.argmax(objectives < current_objective)
        else:
            raise Exception(
                "local_search solution_selector must be one of 'best_improvement', 'first_improvement'"
            )

        permutation = neighbourhood[candidate_idx]
        candidate_objective = objectives[candidate_idx]

        if candidate_objective < current_objective:
            current_objective = candidate_objective
            swap(current_encoding, permutation[0], permutation[1])
        else:
            break

    return current_encoding, current_objective


def multistart(
    flow: np.array,
    distance: np.array,
    neighbourhood_builder: str,
    k: int,
    solution_selector: str,
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
    neighbourhood_builder: str,
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

    if neighbourhood_builder == "total_swap":
        neighbourhood = list(combinations(range(n), 2))
    elif neighbourhood_builder == "adjacent_swap":
        neighbourhood = [(i, (i + 1) % (n)) for i in range(n)]
    else:
        raise Exception(
            "local_search neighbourood_builder must be one of 'total_swap', 'adjacent_swap'"
        )
    
    for j in range(k):
        # initial random feasible solution or given deterministic
        if given_start:
            solution_encoding = deterministic_start[j]
        else:
            solution_encoding = [i for i in range(n)]
            random.shuffle(solution_encoding)
            
        current_objective = calculate_objective(solution_encoding, flow, distance)

        while True:
            (
                candidate_solution,
                candidate_objective,
            ) = calculate_neighbourhood_improving_similar(
                neighbourhood, flow, distance, local_optima, solution_encoding, current_objective
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
    instance_filepath = "../data/qapdata/tai256c.dat"
    flow, distance = read_instance_data(instance_filepath)
    n = len(flow)
    best_obj, _ = read_optimal_solution("../data/qapsoln/tai256c.sln")

    t0 = time.time()
    fast_optimal_local_search_solution, fast_objective = local_search(
        list(range(n)), flow, distance
    )
    t1 = time.time()
    print(f"fast local search took {t1-t0}")
    
    print(
        f"FAST Optimal solution, {fast_optimal_local_search_solution} has objective {fast_objective}"
    )
    print(f"solution: {best_obj}")
