import numpy as np


def find_weighted_mean(values: np.array, errors: np.array):
    weights = 1 / errors**2
    weighted_mean = np.sum(
        values * weights
    ) / np.sum(weights)
    combined_error = np.sqrt(1 / np.sum(weights))
    return weighted_mean, combined_error
