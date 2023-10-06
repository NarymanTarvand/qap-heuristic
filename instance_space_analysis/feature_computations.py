from collections import defaultdict
from itertools import combinations, product
import math

import numpy as np
import pandas as pd

from global_calculations import read_instance_data


def get_problem_size(data_matrix: np.ndarray) -> int:
    """returns problem size, data_matrix can be flow or distance matrix"""
    return len(data_matrix)


def get_matrix_sparsity(data_matrix: np.ndarray) -> float:
    return np.count_nonzero(data_matrix == 0) / len(data_matrix)


def get_matrix_asymmetry(data_matrix: np.ndarray) -> float:
    symmetric_pairs = 0
    for (i, j) in combinations(range(len(data_matrix)), 2):
        if data_matrix[i, j] == data_matrix[j, i]:
            symmetric_pairs += 1
    return symmetric_pairs / math.comb(len(data_matrix), 2)


def get_matrix_dominance(data_matrix: np.ndarray) -> float:
    flat_matrix = data_matrix.flatten()
    if np.mean(flat_matrix) == 0:
        return np.std(flat_matrix)
    else:
        return np.std(flat_matrix) / np.mean(flat_matrix)


def get_matrix_max(data_matrix: np.ndarray) -> float:
    return np.max(data_matrix.flatten())


def get_matrix_min(data_matrix: np.ndarray) -> float:
    return np.min(data_matrix.flatten())


def get_matrix_mean(data_matrix: np.ndarray) -> float:
    return np.mean(data_matrix.flatten())


def get_matrix_median(data_matrix: np.ndarray) -> float:
    return np.median(data_matrix.flatten())


def get_instance_features(data: pd.DataFrame, data_fp: str):
    features = [
        get_problem_size,
        get_matrix_sparsity,
        get_matrix_asymmetry,
        get_matrix_dominance,
        get_matrix_max,
        get_matrix_min,
        get_matrix_mean,
        get_matrix_median,
    ]
    feature_names = [
        (
            "flow_" + feature.__name__.split("_")[-1],
            "distance_" + feature.__name__.split("_")[-1],
        )
        for feature in features
    ]
    feature_values = {}
    for instance in data.index:

        instance_fp = data_fp + instance + ".dat"
        flow, distance = read_instance_data(instance_fp)
        yep2 = product(features, [flow, distance])
        feature_values[instance] = [i(j) for i, j in yep2]

    column_names = [
        "drop",
        "problem_size",
        "flow_sparsity",
        "distance_sparsity",
        "flow_asymmetry",
        "distance_asymmetry",
        "flow_dominance",
        "distance_dominance",
        "flow_max",
        "distance_max",
        "flow_min",
        "distance_min",
        "flow_mean",
        "distance_mean",
        "flow_median",
        "distance_median",
    ]
    feature_df = pd.DataFrame.from_dict(feature_values, orient="index")
    feature_df = feature_df.rename(
        columns={i: column_names[i] for i in feature_df.columns}
    )
    feature_df = feature_df.loc[:, feature_df.columns != "drop"]

    return feature_df
