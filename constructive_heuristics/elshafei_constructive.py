import sys
import warnings

import numpy as np

sys.path.append('..') 

from global_calculations import read_instance_data

# Implements constructive heuristic from [1]
#  |  [1]  Hospital Layout as a Quadratic Assignment Problem, 
#  |       ALWALID N. ELSHAFEI, Journal of Operational Research 

# TODO: do this with generators + move to global_calculations
def calculate_partial_objective(partial_solution_encoding, 
                                distance, flow) -> int:
    
    """ -1 for unallocated facilities """
    cost = 0

    for object_i, location_i in enumerate(partial_solution_encoding):
        if location_i != -1:
            for object_j, location_j in enumerate(partial_solution_encoding):
                if location_j != -1:
                    cost += flow[object_i, object_j] * distance[location_i, location_j]
    
    return cost

def augment_solution(solution, facility_mask, location_mask, 
                     new_facility, new_location) -> None:
    
    solution[new_facility] = new_location
    facility_mask[new_facility] = 1
    location_mask[new_location] = 1

def step1(n, distance) -> list:
    return sorted(range(n), key=lambda x: np.sum(distance[x, :]))

def step2(n, distance, flow) -> list:
    E_map = {}
    for i in range(n):
        number_interactions = np.count_nonzero(flow[:, i]) + \
                              np.count_nonzero(flow[i, :]) - \
                              np.count_nonzero([flow[i, i]])

        total_interactions = np.sum(flow[: ,i]) + np.sum(flow[i, :]) - \
                             flow[i, i]
        E_map[i] = (number_interactions, total_interactions)

    E1_rank = sorted(range(n), key=lambda x: E_map[x][0], reverse=True)
    E2_rank = sorted(range(n), key=lambda x: E_map[x][1], reverse=True)

    return sorted(range(n), key=lambda x: E1_rank.index(x) + E2_rank.index(x))

def step4(facility_rank, location_rank, facility_mask, location_mask) -> tuple:
    for facility in facility_rank:
        if facility_mask[facility] == 0:
            new_facility = facility
            break

    for location in location_rank:
        if location_mask[location] == 0:
            new_location = location
            break

    return new_facility, new_location

def step5(n, new_facility, facility_mask, location_mask, 
          flow, distance, solution):
    
    max_interaction_facility = max([fac for fac in range(n) if facility_mask[fac] == 0],
                                   key=lambda x: flow[x, new_facility] + flow[new_facility, x])
    
    max_obj_delta_location = min([loc for loc in range(n) if location_mask[loc] == 0],
                                 key=lambda x: calculate_partial_objective(solution, distance, flow))
    
    return max_interaction_facility, max_obj_delta_location

def step6(solution, new_facility, new_location, 
          distance, flow, obj) -> None:
    
    swapped_obj = obj
    swap_locations = (x for x in solution if x not in [-1, new_location])
    swapped_solution = solution.copy()
    swap_index = None

    for x in swap_locations:
        swap_facility = np.where(swapped_solution == x)
        swapped_solution[swap_facility], swapped_solution[new_facility] = swapped_solution[new_facility], swapped_solution[swap_facility]
        swapped_obj = calculate_partial_objective(swapped_solution, 
                                                  distance, flow)
        
        swapped_solution[new_facility], swapped_solution[swap_facility] = swapped_solution[swap_facility], swapped_solution[new_facility]

        if swapped_obj < obj:
            obj = swapped_obj
            swap_index = x
    
    if swap_index is not None:
        swapped_facility = np.where(swapped_solution == swap_index)
        solution[swapped_facility], solution[new_facility] = solution[new_facility], solution[swapped_facility]

def elshafei_constructive(distance, flow):
    n = len(distance)
    solution = np.full((1,n), -1)[0]
    facility_mask, location_mask = np.zeros(n), np.zeros(n)

    location_rank, facility_rank = step1(n, distance), step2(n, distance, flow)

    # step 3: check whether any facility remains unassigned
    while (-1 in solution):
        unique, counts = np.unique(solution, return_counts=True)
        print(dict(zip(unique, counts))[-1])

        new_facility, new_location = step4(facility_rank, location_rank, 
                                           facility_mask, location_mask)
        
        augment_solution(solution, facility_mask, location_mask, 
                         new_facility, new_location)
        
        obj = calculate_partial_objective(solution, distance, flow)
        key = 1

        step6(solution, new_facility, new_location, distance, flow, obj)
        if key == 1:
            if -1 not in solution:
                return solution
            else:
                
                max_interaction_facility, max_obj_delta_location = step5(n, new_facility, facility_mask, 
                                                                         location_mask, flow, distance, solution)
                augment_solution(solution, facility_mask, location_mask, 
                                 max_interaction_facility, max_obj_delta_location)
                
                obj = calculate_partial_objective(solution, distance, flow)

                key = 2
                step6(solution, max_interaction_facility, max_obj_delta_location, distance, flow, obj)
                continue
        else:
            continue     
    return solution, calculate_partial_objective(solution, distance, flow)

if __name__ == "__main__":
    warnings.filterwarnings('ignore') # getting weird deprecation issue
    distance, flow = read_instance_data("../../QAPInstances/tai150b.dat")
    solution = elshafei_constructive(distance, flow)
    print(f"found solution: {solution}")
    print(calculate_partial_objective(solution, distance, flow))