from itertools import product
import numpy as np
from collections import defaultdict

def delta_objective(new_facility: int, new_location : int,
                    new_encoding: np.array, 
                    distance: np.array, flow: np.array)  -> int:
    """ calculates a difference in objective given a current encoding and
        a new solution encoding from the constructive heuristic"""

    total_facilities = np.nonzero(new_encoding + 1)[0]
    total_locations = new_encoding[total_facilities]

    return np.dot(distance[[new_location], :][:, total_locations][0], 
                  flow[[new_facility], :][:, total_facilities][0]) + \
                  np.dot(distance[total_locations, :][:, [new_location]].reshape((1,-1))[0], 
                  flow[total_facilities, :][:, [new_facility]].reshape((1,-1))[0])

def greedy_constructive(distance: np.array, flow: np.array) -> np.aray:
    """ iteratively fills a solution vector, minimising change in 
        objective function at each step """

    n = distance.shape[0]

    solution = np.full(n, -1)
    solution_copy = np.full(n, -1)

    available_rows, available_columns = set(range(n)), set(range(n))
    objective = 0

    while not np.all(solution+1):
        curr_min_value = np.inf
        
        for pair in product(available_rows, available_columns):
            solution_copy[pair[0]] = pair[1]
            delta_obj = delta_objective(pair[0], pair[1], solution_copy, 
                                        distance, flow)
            solution_copy[pair[0]] = -1

            if delta_obj < curr_min_value:
               curr_min,  curr_min_value = pair, delta_obj
               if curr_min_value == 0:
                break

        solution[curr_min[0]] = curr_min[1]
        solution_copy[curr_min[0]] = curr_min[1]
        
        # print(curr_min_value)
        objective += curr_min_value

        available_rows.remove(curr_min[0])
        available_columns.remove(curr_min[1])
        
    return solution

def main_constructive(distance: np.array, flow: np.array) -> np.array:
    """ main constructive heuristic """

    # heuristic such that "flow" contain larger values
    if np.sum(distance) > np.sum(flow):
        flow, distance = distance, flow

    n = distance.shape[0]
    solution = np.full(n, -1)

    R = sorted(range(n), key = lambda x: np.sum(np.square(distance[x, :])))
    L = sorted(range(n), key = lambda x: np.sum(np.square(flow[x, :])), 
               reverse = True)

    L_val = [np.sum(np.square(flow[i, :])) for i in L]

    repeats = defaultdict(list)
    for i, j in enumerate(L):
        repeats[L_val[i]].append(j)

    L_tie = []
    for i in repeats.values():
        L_tie += sorted(i, key = lambda x: np.sum(np.square(flow[:, x])), 
                        reverse = True)

    for i in range(n):
        solution[L_tie[i]] = R[i]
        
    return solution

def hospital_constructive(distance: np.array, flow: np.array) -> np.array:
    n = distance.shape[0]
    solution = np.full(n, -1)

    R = sorted(range(n), key = lambda x: np.sum(distance[x, :]))

    E_n = sorted(range(n), key = lambda x: len(np.nonzero(flow[:, x])), 
                 reverse=True)
    E_int = sorted(range(n), key = lambda x: np.sum(flow[:, x]) + np.sum(flow[x ,:]), reverse=True)

    L_map = defaultdict(int)
    for i in range(n):
        L_map[i] = E_n.index(i) + E_int.index(i)
    L = sorted(range(n), key = lambda x: L_map[x])
    
    for i in range(n):
        solution[L[i]] = R[i]

    #TODO: Finish this
    
    return solution