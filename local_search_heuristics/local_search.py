from itertools import combinations
from typing import Callable

import numpy as np

from global_calculations import (
    calculate_objective,
    read_instance_data,
    read_optimal_solution,
)

from local_search_heuristics.selection import *
from local_search_heuristics.neighbours import *

def greedy_local_search(
    solution_encoding: list, flow: np.array, distance: np.array,
    neighbourhood_builder: Callable
) -> list:

    current_objective = calculate_objective(solution_encoding, flow, distance)
    n = len(flow)

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
        solution_encoding, flow, distance, get_adjacent_swap_neighbourhood
    )
    print(
        f"Optimal solution, {optimal_local_search_solution} has objective {objective}"
    )
