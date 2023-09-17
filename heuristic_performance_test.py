import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from global_calculations import read_instance_data, read_optimal_solution
from local_search_heuristics.greedy_one_step_local import greedy_one_step_local


def record_heuristic_performance(
    data_directory: str = "data/qapdata/",
    solution_directory: str = "data/qapsoln/",
    output_filepath: str = "data/heuristic_performance.csv",
):
    instance_record = {
        "instance": [],
        "solution_objective": [],
        "local_search_objective": [],
    }
    pbar = tqdm(os.listdir(data_directory))
    for filename in pbar:
        pbar.set_description(f"Processing {filename}")
        optimal_solution_filepath = solution_directory + filename.split(".")[0] + ".sln"
        if not os.path.exists(optimal_solution_filepath):
            continue
        instance_record["instance"].append(filename.split(".")[0])
        optimal_objective, optimal_encoding = read_optimal_solution(
            optimal_solution_filepath
        )
        instance_record["solution_objective"].append(optimal_objective)

        flow, distance = read_instance_data(data_directory + filename)

        solution_encoding = np.random.permutation(
            len(flow)
        )  # TODO: come up with a non-random initial solution
        local_search_encoding, local_search_objective = greedy_one_step_local(
            solution_encoding, flow, distance
        )
        instance_record["local_search_objective"].append(local_search_objective)

        # TODO: Add other heuristics

    pd.DataFrame(instance_record).to_csv(output_filepath)


if __name__ == "__main__":
    record_heuristic_performance()
