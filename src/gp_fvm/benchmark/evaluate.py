import numpy as np
from .hyperopt import find_lengthscales

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


def evaluate_params(
    params: list[any],
    lengthscale_grid: list[list[np.floating]],
    conditioning_fn: callable,
    error_estimator: callable,
    *,
    use_tqdm=True,
    notebook=True,
):
    lengthscales = []
    errors = []
    if use_tqdm:
        if notebook:
            params = tqdm_notebook(params)
        else:
            params = tqdm(params)
    for param in params:
        def conditioning_fn_lengthscales(*lengthscales):
            return conditioning_fn(param, *lengthscales)

        l = find_lengthscales(
            lengthscale_grid, conditioning_fn_lengthscales, error_estimator
        )
        lengthscales.append(l)
        errors.append(error_estimator(conditioning_fn(param, *l)))
        if use_tqdm:
            params.set_description(f"Param: {param}, lengthscales: {l} , error: {errors[-1]:.2e}")
    return lengthscales, errors
