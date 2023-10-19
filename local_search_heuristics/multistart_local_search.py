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
from local_search_heuristics.local_search import local_search

from local_search_heuristics.selection import *
from local_search_heuristics.neighbours import *


def multistart(
    flow: np.array,
    distance: np.array,
    neighbourhood_builder: str,
    k: int,
    solution_selector: str,
    deterministic_start: list = [],
    restrict_time: bool = False,
) -> Tuple[List[int], float]:
    multistart_solutions = []
    multistart_costs = []
    n = len(flow)

    given_start = False
    if deterministic_start != []:
        given_start = True
        assert len(deterministic_start) == k

    t0 = time.time()
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
    multistart_solution = multistart_solutions[np.argmin(multistart_costs)]

    if restrict_time and (time.time() - t0 > 60):
        return multistart_solution, multistart_cost

    return multistart_solution, multistart_cost


if __name__ == "__main__":
    if len(sys.argv) != 4:
        logging.error("Error: expected 3 arg")
        logging.warning(
            "Usage: python -m local_search_heuristics.local_search <instance_name> <restricted_time> <n_multistart>"
        )
        sys.exit(1)
    instance_name = sys.argv[1]
    restricted_time = sys.argv[2]
    n_multistart = int(sys.argv[3])

    instance_filepath = f"data/qapdata/{instance_name}.dat"
    optimal_solution_filepath = f"data/qapsoln/{instance_name}.sln"
    flow, distance = read_instance_data(instance_filepath)
    optimal_objective, optimal_encoding = read_optimal_solution(
        optimal_solution_filepath
    )

    deterministic_starts = [
        np.random.permutation(len(flow)) for _ in range(n_multistart)
    ]
    start_time = time.time()
    solution, objective = multistart(
        flow,
        distance,
        "total_swap",
        n_multistart,
        "best_improvement",
        deterministic_start=deterministic_starts,
        restrict_time=restricted_time,
    )

    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"Heuristic solution, {solution} has objective {objective}")
    print(f"Optimal solution, {optimal_encoding} has objective {optimal_objective}")
