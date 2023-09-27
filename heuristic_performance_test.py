import os
from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import time

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
    (5, get_adjacent_swap_neighbourhood),
    (5, get_total_swap_neighbourhood),
    (10, get_adjacent_swap_neighbourhood),
    (10, get_total_swap_neighbourhood),
    (25, get_adjacent_swap_neighbourhood),
    (25, get_total_swap_neighbourhood),
    (50, get_adjacent_swap_neighbourhood),
    (100, get_adjacent_swap_neighbourhood),
]


def record_heuristic_performance(
    data_directory: str = "data/qapdata/",
    solution_directory: str = "data/qapsoln/",
    output_filepath: str = "data/heuristic_performance.csv",
) -> None:

    # TODO: remove print and time stuff

    records = defaultdict(list)

    pbar = tqdm(os.listdir(data_directory)[:2])
    for filename in pbar:
        pbar.set_description(f"Processing {filename}")
        optimal_solution_filepath = solution_directory + filename.split(".")[0] + ".sln"

        if not os.path.exists(optimal_solution_filepath):
            continue

        records["instance"].append(filename.split(".")[0])
        optimal_objective, optimal_encoding = read_optimal_solution(
            optimal_solution_filepath
        )
        records["optimal_objective"].append(optimal_objective)
        records["optimal_solution"].append(optimal_encoding)

        flow, distance = read_instance_data(data_directory + filename)
        n = len(flow)

        t0 = time.time()
        # Constructive Heuristics (+ descent, deterministic) Experiments
        elshafei_solution, elshafei_constructive_obj = elshafei_constructive(
            distance, flow
        )
        records["elshafei_constructive_objective"].append(elshafei_constructive_obj)
        records["elshafei_constructive_solution"].append(elshafei_solution)

        elshafei_descent_solution, elshafei_descent_obj = local_search(
            elshafei_solution,
            flow,
            distance,
            get_total_swap_neighbourhood,
            calculate_neighbourhood_optimal_solution,
        )
        t1 = time.time()
        print(f"elshafei constructive local search took {t1-t0}")
        records["elshafei_constructive_greedy_local_search_objective"].append(
            elshafei_descent_obj
        )
        records["elshafei_constructive_greedy_local_search_solution"].append(
            elshafei_descent_solution
        )

        t0 = time.time()
        constructive_solution, constructive_obj = main_constructive(distance, flow)
        records["constructive_objective"].append(constructive_obj)
        records["constructive_solution"].append(constructive_solution)

        constructive_descent_solution, constructive_descent_obj = local_search(
            constructive_solution,
            flow,
            distance,
            get_total_swap_neighbourhood,
            calculate_neighbourhood_optimal_solution,
        )
        t1 = time.time()
        print(f"main constructive local search took {t1-t0}")

        records["constructive_greedy_local_search_objective"].append(
            constructive_descent_obj
        )
        records["constructive_greedy_local_search_solution"].append(
            constructive_descent_solution
        )

        # Dissimilarity Comparison Experiment (vs. Multistart)
        for k, builder in MULTISTART_PARAMETERS:
            t0 = time.time()
            deterministic_starts = [np.random.permutation(n) for _ in range(k)]
            multistart_solution, multistart_obj = multistart(
                flow,
                distance,
                builder,
                k,
                calculate_neighbourhood_optimal_solution,
                deterministic_start=deterministic_starts,
            )
            (
                first_improvement_multistart_solution,
                first_improvement_multistart_obj,
            ) = multistart(
                flow,
                distance,
                builder,
                k,
                calculate_neighbourhood_first_improvement,
                deterministic_start=deterministic_starts,
            )
            dissimilarity_solution, dissimilarity_obj = disimilarity_local_search(
                flow, distance, builder, k, deterministic_start=deterministic_starts
            )
            t1 = time.time()
            print(
                f"multistart and dissimilarity with k = {k}, builder = {builder} took {t1-t0}"
            )

            records[
                f"{k}_multistart_{builder.__name__.replace('get_', '')}_optimal_neighbour_objective"
            ].append(multistart_obj)
            records[
                f"{k}_multistart_{builder.__name__.replace('get_', '')}_optimal_neighbour_solution"
            ].append(multistart_solution)

            records[
                f"{k}_multistart_{builder.__name__.replace('get_', '')}_first_improvement_objective"
            ].append(first_improvement_multistart_obj)
            records[
                f"{k}_multistart_{builder.__name__.replace('get_', '')}_first_improvement_solution"
            ].append(first_improvement_multistart_solution)

            records[
                f"{k}_dissimilarity_local_search_{builder.__name__.replace('get_', '')}_objective"
            ].append(dissimilarity_obj)
            records[
                f"{k}_dissimilarity_local_search_{builder.__name__.replace('get_', '')}_solution"
            ].append(dissimilarity_solution)

        # GRASP (+ local search, simulated annealing) Experiments
        t0 = time.time()
        grasp_search_methods = ["local search", "simulated annealing"]
        for search_method in grasp_search_methods:
            _, grasp_obj = randomised_greedy_grasp(
                n_iterations=25,
                flow=flow,
                distance=distance,
                search_method=search_method,
            )

            records[f"grasp_{search_method.replace(' ', '_')}"].append(grasp_obj)

        t1 = time.time()
        print(f"GRASP took {t1-t0}")

        # Genetic Algorithm Experiments
        t0 = time.time()
        _, genetic_algorithm_obj = genetic_algorithm(
            n,
            N_pop=100,
            distance_data=distance,
            flow_data=flow,
            MaxIter=150,
            p_crossover=0.8,
            p_mutation=0.2,
        )

        records["genetic_algorithm"].append(genetic_algorithm_obj)
        t1 = time.time()
        print(f"Genetic took {t1-t0}")

    pd.DataFrame(records).to_csv(output_filepath)


if __name__ == "__main__":
    record_heuristic_performance()
