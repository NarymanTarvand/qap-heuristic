from math import comb
from itertools import combinations
from random import shuffle

import numpy as np

def swap(arr, idx1, idx2) -> None:
    """ swaps elements in array given 2 indexes"""
    arr[idx1], arr[idx2] = arr[idx2], arr[idx1]

def get_adjacent_swap_neighbourhood(solution_encoding: list, n: int) -> list:
    neighbourhood = []
    for idx, _ in enumerate(solution_encoding):
        neighbour = solution_encoding.copy()
        swap(neighbour, idx, (idx + 1) % n)
        neighbourhood.append(neighbour)
    return neighbourhood

def adjacent_swap_generator(solution_encoding: list, n: int):
    idx = 0
    neighbour = solution_encoding.copy()
    while idx < n:
        swap(neighbour, idx, (idx + 1) % n)
        yield neighbour
        swap(neighbour, (idx + 1) % n, idx)
        idx += 1

def get_total_swap_neighbourhood(solution_encoding: list, n: int) -> list:
    """ generates n choose 2 neighbouring swaps """
    neighbourhood = []
    for idx1, idx2 in combinations(range(n), 2):
        neighbour = solution_encoding.copy()
        swap(neighbour, idx1, idx2)
        neighbourhood.append(neighbour)
    return neighbourhood

def total_swap_generator(solution_encoding: list, n: int):
    i = 0
    swap_indexes = combinations(range(n), 2)
    n_swap_indexes = comb(n, 2)
    
    neighbour = solution_encoding.copy()
    
    while i < n_swap_indexes:
        curr_swap_index = next(swap_indexes)
        swap(neighbour, curr_swap_index[0], curr_swap_index[1])
        yield neighbour
        
        swap(neighbour, curr_swap_index[1], curr_swap_index[0])
        i += 1

def append_random_solutions(neighbourhood: list, k: int) -> None:
    for _ in range(k):
        neighbourhood.append(np.random.permutation(k).tolist())

def shuffle_neighbourhood(neighbourhood: list) -> None:
    shuffle(neighbourhood)

if __name__ == "__main__":
    pass
