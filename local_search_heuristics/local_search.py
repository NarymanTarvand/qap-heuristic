import logging
import random
from typing import Tuple, List, Callable

import numpy as np
import time

from global_calculations import (
    calculate_objective,
    calculate_objective_vectorised,
    read_instance_data,
    read_optimal_solution,
)

from local_search_heuristics.selection import *
from local_search_heuristics.neighbours import *


def local_search(
    solution_encoding: list,
    flow: np.array,
    distance: np.array,
    neighbourhood_builder: str = "total_swap",
    solution_selector: str = "best_improvement",
    restrict_time: bool = False,
    logging: bool = False
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
    
    if logging:
        N_iter = 0
        iter_sols = list()
    
    t0 = time.time()
    while True:
        objectives = [
            calculate_objective_vectorised(
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
            if logging:
                print(N_iter)
                N_iter += 1
                iter_sols.append(current_objective)
        else:
            break
        
        if logging and N_iter > 250:
            return iter_sols
        
        if restrict_time and (time.time() - t0 > 60):
            return current_encoding, current_objective
    return current_encoding, current_objective


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Error: expected 2 arg")
        logging.warning(
            "Usage: python -m local_search_heuristics.local_search <instance_name> <restricted_time> <n_multistart>"
        )
        sys.exit(1)
    instance_name = sys.argv[1]
    restricted_time = sys.argv[2]

    instance_filepath = f"data/qapdata/{instance_name}.dat"
    optimal_solution_filepath = f"data/qapsoln/{instance_name}.sln"
    flow, distance = read_instance_data(instance_filepath)
    optimal_objective, optimal_encoding = read_optimal_solution(
        optimal_solution_filepath
    )

    start_time = time.time()
    solution, objective = local_search(
        list(range(len(flow))), flow, distance, restrict_time=restricted_time
    )

    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"Heuristic solution, {solution} has objective {objective}")
    print(f"Optimal solution, {optimal_encoding} has objective {optimal_objective}")
