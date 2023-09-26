import os
from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from global_calculations import read_instance_data, read_optimal_solution

from constructive_heuristics.constructive import (
    main_constructive
)
from constructive_heuristics.elshafei_constructive import (
    elshafei_constructive
)
from local_search_heuristics.neighbours import (
    get_total_swap_neighbourhood, get_adjacent_swap_neighbourhood
)
from local_search_heuristics.selection import (
    calculate_neighbourhood_optimal_solution, 
    calculate_neighbourhood_first_improvement
)
from local_search_heuristics.local_search import (
    local_search, multistart, disimilarity_local_search
)
from metaheuristics.genetic_algorithm import (
    genetic_algorithm
)
from metaheuristics.GRASP.randomised_greedy_grasp import (
    randomised_greedy_grasp
)

def record_heuristic_performance(
    data_directory: str = "data/qapchr/",
    solution_directory: str = "data/qapsoln/",
    output_filepath: str = "data/heuristic_performance.csv",
) -> None:
    
    records = defaultdict(list)
    
    pbar = tqdm(os.listdir(data_directory))
    for filename in pbar:
        pbar.set_description(f"Processing {filename}")
        optimal_solution_filepath = solution_directory + filename.split(".")[0] + ".sln"
        
        if not os.path.exists(optimal_solution_filepath):
            continue
        
        records["instance"].append(filename.split(".")[0])
        optimal_objective, optimal_encoding = read_optimal_solution(
            optimal_solution_filepath
        )
        records["solution_objective"].append(optimal_objective)

        flow, distance = read_instance_data(data_directory + filename)
        n = len(flow)
        
        # Constructive Heuristics (+ descent, deterministic) Experiments
        elshafei_solution, elshafei_constructive_obj = elshafei_constructive(distance, flow)
        
        _, elshafei_descent_obj = local_search(elshafei_solution, 
                                               flow, distance, 
                                               get_total_swap_neighbourhood, 
                                               calculate_neighbourhood_optimal_solution
                                               )
        
        records["elshafei_constructive"].append(elshafei_constructive_obj)
        records["elshafei_constructive+greedy_local_search_total"].append(elshafei_descent_obj)
        
        constructive_solution, constructive_obj = main_constructive(distance, flow)
        
        _, constructive_descent_obj = local_search(constructive_solution, 
                                                   flow, distance, 
                                                   get_total_swap_neighbourhood, 
                                                   calculate_neighbourhood_optimal_solution
                                                   )
    
        records["constructive"].append(constructive_obj)
        records["constructive+greedy_local_search_total"].append(constructive_descent_obj)
        
        # Multistart Local Search Experiments
        multistart_k_list = [5, 10, 25, 50, 75, 100, 120]
        builders = [get_total_swap_neighbourhood, get_adjacent_swap_neighbourhood]
        selectors = [calculate_neighbourhood_optimal_solution, calculate_neighbourhood_first_improvement]
        
        for k, builder, selector in product(multistart_k_list, builders, selectors):
            _, multistart_obj = multistart(flow, distance, builder, k, selector)
            records[f"{k}_multistart_{builder.__name__.replace('get_', '')}_{selector.__name__.replace('calculate_neighbourhood_', '')}"].append(multistart_obj)
        
        # Dissimilarity Comparison Experiment (vs. Multistart)
        comparison_k_list = [5, 10, 15, 20, 25, 30] # can change this
        for k in comparison_k_list:
            deterministic_starts = [np.random.permutation(n) for _ in range(k)]
            _, multistart_obj = multistart(flow, distance, get_total_swap_neighbourhood, k, calculate_neighbourhood_optimal_solution, deterministic_start=deterministic_starts)
            _, dissimilarity_obj = disimilarity_local_search(flow, distance, get_total_swap_neighbourhood, k, deterministic_start=deterministic_starts)
            records[f"{k}_multistart"].append(multistart_obj)
            records[f"{k}_dissimilarity"].append(dissimilarity_obj)
        
        # GRASP (+ local search, simulated annealing) Experiments
        grasp_search_methods = ["local search", "simulated annealing"]
        for search_method in grasp_search_methods:
            _, grasp_obj = randomised_greedy_grasp(n_iterations=25,
                                                   flow=flow, 
                                                   distance=distance,
                                                   search_method=search_method
                                                   )
            
            records[f"grasp_{search_method.replace(' ', '_')}"].append(grasp_obj)
            
        # Genetic Algorithm Experiments
        _, genetic_algorithm_obj = genetic_algorithm(n, N_pop=100, 
                                                     distance_data=distance, 
                                                     flow_data=flow, 
                                                     MaxIter=150, 
                                                     p_crossover=0.8, 
                                                     p_mutation=0.2
                                                     )
        
        records["genetic_algorithm"].append(genetic_algorithm_obj)
        
    pd.DataFrame(records).to_csv(output_filepath)

if __name__ == "__main__":
    record_heuristic_performance()
