import numpy as np
import itertools


def find_lengthscales(
    grids: list[list[np.floating]], conditioning_fn: callable, error_estimator: callable
):
    error_min = np.inf
    l_min = None
    for grid_point in itertools.product(*grids):
        posterior = conditioning_fn(*grid_point)
        error = error_estimator(posterior)
        if error < error_min:
            error_min = error
            l_min = grid_point
    return l_min
