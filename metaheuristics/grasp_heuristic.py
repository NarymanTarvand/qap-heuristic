import numpy as np
from global_calculations import (
    calculate_objective,
    read_instance_data,
    read_optimal_solution,
)
from local_search_heuristics.greedy_one_step_local import greedy_one_step_local


def generate_candidate_list(n_candidates: int, solution_size: int) -> list:
    candidates = []
    for _ in range(n_candidates):
        candidate_solution = np.random.permutation(solution_size)
        candidates.append(candidate_solution)
    return candidates


def calculate_restricted_candidate_list(
    candidate_list: list, threshold: float, flow, distance
) -> list:
    restricted_candidates = []
    for candidate in candidate_list:
        objective = calculate_objective(candidate, flow, distance)
        if objective < threshold:
            restricted_candidates.append((objective, candidate))

    restricted_candidate_list = [x for _, x in sorted(restricted_candidates)]

    return restricted_candidate_list


def random_candidate_selection(
    restricted_candidate_list: list, bias_function: str
) -> list:
    if bias_function not in ["random", "linear", "log", "exponential", "polynomial"]:
        raise Exception(
            "bias_function must be one of 'random', 'linear', 'log', 'exponential', 'polynomial'"
        )

    candidate_ranks = [i + 1 for i, _ in enumerate(restricted_candidate_list)]
    if bias_function == "random":
        bias = [1 for rank in candidate_ranks]
    elif bias_function == "linear":
        bias = [1 / rank for rank in candidate_ranks]
    elif bias_function == "log":
        bias = [1 / np.log(rank + 1) for rank in candidate_ranks]
    elif bias_function == "exponential":
        bias = [np.exp(-rank) for rank in candidate_ranks]
    elif bias_function == "polynomial":
        bias = [-(rank ** (-2)) for rank in candidate_ranks]  # TODO: fix magic num 2

    selection_probability = bias / np.sum(bias)
    solution = restricted_candidate_list[
        np.random.choice(len(restricted_candidate_list), p=selection_probability)
    ]

    return solution


def grasp_heuristic(
    n_iterations: int,
    n_candidates: int,
    threshold: float,
    bias_function: str,
    flow: np.array,
    distance: np.array,
):
    current_objective = np.inf
    for _ in range(n_iterations):
        candidate_list = generate_candidate_list(n_candidates, solution_size=len(flow))
        restricted_candidate_list = calculate_restricted_candidate_list(
            candidate_list, threshold, flow, distance
        )
        candidate_solution = random_candidate_selection(
            restricted_candidate_list, bias_function
        )

        local_search_solution, local_search_objective = greedy_one_step_local(
            candidate_solution, flow, distance
        )

        if local_search_objective < current_objective:
            current_objective = local_search_objective
            current_solution = local_search_solution

    return current_solution, current_objective


if __name__ == "__main__":
    instance_filepath = "data/qapdata/tai30b.dat"
    optimal_solution_filepath = "data/qapsoln/tai30b.sln"
    flow, distance = read_instance_data(instance_filepath)
    optimal_objective, optimal_encoding = read_optimal_solution(
        optimal_solution_filepath
    )

    solution_encoding = [i for i in range(len(flow))]

    read_optimal_solution("data/qapsoln/tai30b.sln")
    optimal_local_search_solution, objective = grasp_heuristic(
        n_iterations=100,
        n_candidates=10,
        threshold=10**10,
        bias_function="random",
        flow=flow,
        distance=distance,
    )
    print(
        f"Heuristic solution, {optimal_local_search_solution} has objective {objective}"
    )
    print(f"Optimal solution, {optimal_encoding} has objective {optimal_objective}")
