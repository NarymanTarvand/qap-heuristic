import os
from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from joblib import Parallel, delayed

from global_calculations import read_instance_data, read_optimal_solution

from constructive_heuristics.constructive import main_constructive
from constructive_heuristics.elshafei_constructive import elshafei_constructive
from local_search_heuristics.neighbours import (
    get_total_swap_neighbourhood,
    get_adjacent_swap_neighbourhood,
)
from local_search_heuristics.selection import (
    calculate_neighbourhood_optimal_solution,
    calculate_neighbourhood_first_improvement,
)
from local_search_heuristics.local_search import (
    local_search,
    multistart,
    disimilarity_local_search,
)
from metaheuristics.genetic_algorithm import genetic_algorithm
from metaheuristics.GRASP.randomised_greedy_grasp import randomised_greedy_grasp

MULTISTART_PARAMETERS = [
    (5, "adjacent_swap"),
    (5, "total_swap"),
    (10, "adjacent_swap"),
    (10, "total_swap"),
    (25, "adjacent_swap"),
    (25, "total_swap"),
    # (50, get_adjacent_swap_neighbourhood),
    # (100, get_adjacent_swap_neighbourhood),
]

# TODO: add time recording to instances.
@delayed
def record_heuristic_performance(
    idx: int,
    instance_name: str,
    data_directory: str = "data/qapdata/",
    solution_directory: str = "data/qapsoln/",
) -> dict:

    print(f"{idx}: {instance_name}")

    performance_record = {}

    optimal_solution_filepath = (
        solution_directory + instance_name.split(".")[0] + ".sln"
    )
    optimal_objective, _ = read_optimal_solution(optimal_solution_filepath)

    performance_record["instance"] = instance_name.split(".")[0]
    performance_record["optimal_objective"] = optimal_objective

    flow, distance = read_instance_data(data_directory + instance_name)
    n = len(flow)

    # Constructive Heuristics (+ descent, deterministic)
    elshafei_solution, elshafei_constructive_obj = elshafei_constructive(distance, flow)
    _, elshafei_descent_obj = local_search(elshafei_solution, flow, distance)

    performance_record["elshafei_constructive_objective"] = elshafei_constructive_obj
    performance_record[
        "elshafei_constructive_greedy_local_search_objective"
    ] = elshafei_descent_obj

    # standard constructive heuristic
    constructive_solution, constructive_obj = main_constructive(distance, flow)
    _, constructive_descent_obj = local_search(
        constructive_solution,
        flow,
        distance,
    )

    performance_record["constructive_objective"] = constructive_obj
    performance_record[
        "constructive_greedy_local_search_objective"
    ] = constructive_descent_obj

    # Dissimilarity Comparison Experiment (vs. Multistart)
    for k, builder in MULTISTART_PARAMETERS:
        deterministic_starts = [np.random.permutation(n) for _ in range(k)]

        _, multistart_obj = multistart(
            flow,
            distance,
            builder,
            k,
            "best_improvement",
            deterministic_start=deterministic_starts,
        )
        _, FI_multistart_obj = multistart(
            flow,
            distance,
            builder,
            k,
            "first_improvement",
            deterministic_start=deterministic_starts,
        )
        _, dissimilarity_obj = disimilarity_local_search(
            flow, distance, builder, k, deterministic_start=deterministic_starts
        )

        performance_record[
            f"{k}_multistart_{builder}_optimal_neighbour_objective"
        ] = multistart_obj

        performance_record[
            f"{k}_multistart_{builder}_first_improvement_objective"
        ] = FI_multistart_obj

        performance_record[
            f"{k}_dissimilarity_local_search_{builder}_objective"
        ] = dissimilarity_obj

    # GRASP (local search, simulated annealing as descent methods)
    grasp_search_methods = ["local search", "simulated annealing"]
    for search_method in grasp_search_methods:
        _, grasp_obj = randomised_greedy_grasp(
            n_iterations=25,
            flow=flow,
            distance=distance,
            search_method=search_method,
        )

        performance_record[f"grasp_{search_method.replace(' ', '_')}"] = grasp_obj

    # Genetic Algorithm
    _, genetic_algorithm_obj = genetic_algorithm(
        n,
        N_pop=100,
        distance_data=distance,
        flow_data=flow,
        MaxIter=150,
        p_crossover=0.8,
        p_mutation=0.2,
    )
    performance_record["genetic_algorithm"] = genetic_algorithm_obj

    return performance_record


def record_heuristics_in_parallel(
    data_directory: str = "data/qapdata/",
    solution_directory: str = "data/qapsoln/",
    output_filepath: str = "data/heuristic_performance.csv",
) -> None:

    valid_instance_names = [
        filename
        for filename in os.listdir(data_directory)
        if os.path.exists(solution_directory + filename.split(".")[0] + ".sln")
    ]

    records = Parallel(n_jobs=-1, verbose=10, prefer="processes")(
        record_heuristic_performance(
            idx,
            instance_name,
            data_directory,
            solution_directory,
        )
        for idx, instance_name in enumerate(valid_instance_names)
    )

    pd.DataFrame(records).to_csv(output_filepath, index=False)


if __name__ == "__main__":
    record_heuristics_in_parallel()
