from collections import defaultdict
from itertools import product
import time
import logging

import numpy as np

from global_calculations import (
    read_instance_data,
    read_optimal_solution,
)

from local_search_heuristics.local_search import local_search
from local_search_heuristics.neighbours import get_total_swap_neighbourhood
from metaheuristics.simulated_annealing import simmulated_annealing
import sys


def greedy_randomised(
    flow: np.array,
    distance: np.array,
):
    """
    Produce a greedy randomised solution for Quadratic Assignment Problem.
    First, define a list of costs that are the largest possible flow paired with the smallest distance.
    Then randomly select from the list of the list of costs, weighted by cost value.
    These indexes give you where the first two factories should be assigned.

    For the rest of the factories, consider the additional cost of that assignment to a possible location.
    Randomly select from the lowest cost options
    """
    n = len(flow)
    assigned_facilities = []
    assigned_locations = []
    unassigned_facilities = [i for i in range(n)]
    unassigned_locations = [i for i in range(n)]
    candidate_soln = [-1 for _ in range(n)]

    distance_matrix = distance + np.transpose(distance)
    flow_matrix = flow + np.transpose(flow)

    # get upper triangular index as these get us all possible flow and distance values.
    triu_idx = np.triu_indices(n, k=1)

    triu_coordinates = [
        (triu_idx[0][idx], triu_idx[1][idx]) for idx in range(len(triu_idx[0]))
    ]
    sorted_distance_idx = [
        x for _, x in sorted(zip(distance_matrix[triu_idx], triu_coordinates))
    ]
    sorted_flow_idx = [
        x for _, x in sorted(zip(flow_matrix[triu_idx], triu_coordinates), reverse=True)
    ]

    # define costs as the smallest distance multiplied by the largest flow.
    sorted_distance = sorted(distance_matrix[triu_idx])
    sorted_flow = sorted(flow_matrix[triu_idx], reverse=True)
    costs = [a * b for a, b in zip(sorted_distance, sorted_flow)]

    # if 0 is in costs, randomly select a zero cost index,
    # otherwise select a cost, weighted by the cost value
    if 0 in costs:
        zero_cost_idx = [i for i, cost in enumerate(costs) if cost == 0]
        candidate_idx = np.random.choice(zero_cost_idx)
    else:
        prob = costs / sum(costs)
        candidate_idx = np.random.choice([i for i, _ in enumerate(costs)], p=prob)

    facility = sorted_distance_idx[candidate_idx]
    location = sorted_flow_idx[candidate_idx]

    candidate_soln[location[0]] = facility[0]
    candidate_soln[location[1]] = facility[1]

    assigned_facilities.append(
        unassigned_facilities.pop(unassigned_facilities.index(facility[0]))
    )
    assigned_facilities.append(
        unassigned_facilities.pop(unassigned_facilities.index(facility[1]))
    )

    assigned_locations.append(
        unassigned_locations.pop(unassigned_locations.index(location[0]))
    )
    assigned_locations.append(
        unassigned_locations.pop(unassigned_locations.index(location[1]))
    )

    # while you have unassigned factories, assign them based on the smallest additional cost.
    while -1 in candidate_soln:
        cost_dict = defaultdict()
        for location_i, facility_k in product(
            unassigned_locations, unassigned_facilities
        ):
            cost_dict[(location_i, facility_k)] = np.sum(
                flow[location_i, assigned_locations][:, np.newaxis]
                * distance[facility_k, assigned_facilities][np.newaxis, :]
            )

        random_low_cost = np.random.choice(
            sorted(cost_dict.values())[: len(unassigned_locations) // 2 + 1]
        )

        min_location, min_facility = list(cost_dict.keys())[
            list(cost_dict.values()).index(random_low_cost)
        ]

        unassigned_locations.remove(min_location)
        assigned_locations.append(min_location)

        unassigned_facilities.remove(min_facility)
        assigned_facilities.append(min_facility)

        candidate_soln[min_location] = min_facility

    return candidate_soln


def randomised_greedy_grasp(
    n_iterations: int,
    flow: np.array,
    distance: np.array,
    search_method: str,
    restrict_time: bool = True,
):
    """
    Randomised Greedy GRASP heuristic for Quadratic Assignment Problem.
    First generate a greedy randomised solution.
    Then use either local search or simulated annealing to improve the solution.
    """
    current_objective = np.inf

    t0 = time.time()
    for i in range(n_iterations):
        candidate_solution = greedy_randomised(flow, distance)
        if search_method == "local search":
            local_descent_solution, local_descent_objective = local_search(
                candidate_solution, flow, distance
            )
        elif search_method == "simulated annealing":
            local_descent_solution, local_descent_objective = simmulated_annealing(
                candidate_solution, flow, distance
            )
        else:
            raise Exception(
                "search_method must be one of 'local search', 'simulated annealing'"
            )

        if local_descent_objective < current_objective:
            current_objective = local_descent_objective
            current_solution = local_descent_solution

        if restrict_time and (time.time() - t0 > 60):
            return current_solution, current_objective

    return current_solution, current_objective


if __name__ == "__main__":
    if len(sys.argv) != 4:
        logging.error("Error: expected 3 arg")
        logging.warning(
            "Usage: python -m metaheuristics.GRASP.randomised_greedy_grasp <instance_name> <restricted_time> <search_method>"
        )
        sys.exit(1)
    instance_name = sys.argv[1]
    restricted_time = sys.argv[2]
    search_method = sys.argv[3]

    instance_filepath = f"data/qapdata/{instance_name}.dat"
    optimal_solution_filepath = f"data/qapsoln/{instance_name}.sln"
    flow, distance = read_instance_data(instance_filepath)
    optimal_objective, optimal_encoding = read_optimal_solution(
        optimal_solution_filepath
    )
    start_time = time.time()
    optimal_randomised_greedy_grasp_solution, objective = randomised_greedy_grasp(
        n_iterations=10,
        flow=flow,
        distance=distance,
        search_method=search_method,
        restrict_time=restricted_time,
    )
    print("--- %s seconds ---" % (time.time() - start_time))

    print(
        f"Heuristic solution, {optimal_randomised_greedy_grasp_solution} has objective {objective}"
    )
    print(f"Optimal solution, {optimal_encoding} has objective {optimal_objective}")
