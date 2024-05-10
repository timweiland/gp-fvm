import argparse
import linpde_gp
import numpy as np
import probnum as pn
from gp_fvm.utils.figure_manager import FigureManager
from gp_fvm.plot.dim1 import plot_function, compare_to_solution_gp, plot_gp

from darcy_utils import read_data, fit_col

from matplotlib import pyplot as plt

import pandas as pd

from pathlib import Path
import uuid
from tqdm import tqdm
import h5py

import optuna
import yaml

def read_data(filename):
    return h5py.File(filename)

def main():
    # l_t, l_x, hf, problem_idx, beta, N_pde_t=80, N_pde_x=80, ic_stride=10, N_bc=50, output_scale=1.0, 
    parser = argparse.ArgumentParser()
    parser.add_argument('--l-xy', type=float, default=0.2)
    parser.add_argument('--hdf', type=str, default='../data/2D_DarcyFlow_beta1.0_Train.hdf5')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--N-bc', type=int, default=30)
    parser.add_argument('--N-pde-xy', type=int, default=10)
    parser.add_argument('--results-path', type=str, default="./results/darcy")
    args = parser.parse_args()

    hf = read_data(args.hdf)

    N_problems_search = 10

    def hyperopt_objective(trial):
        l_xy = trial.suggest_float('l_xy', 1e-4, 3.0)
        mean_MSE = 0.0
        for i in range(N_problems_search):
            errs, _ = fit_col(
                l_xy,
                hf,
                i,
                beta=args.beta,
                N_bc=args.N_bc,
                N_xy=args.N_pde_xy
            )
            mean_MSE += errs[1]
        mean_MSE /= N_problems_search
        return mean_MSE

    print("Optimizing hyperparameters...")
    study = optuna.create_study(sampler=optuna.samplers.GPSampler())
    study.optimize(hyperopt_objective, n_trials=30)
    print(f"Best trial: {study.best_trial}")
    min_l_xy = study.best_params['l_xy']

    N_problems = hf.get('tensor').shape[0]
    L_inf_errs = []
    MSEs = []
    runtimes = []
    for i in range(N_problems):
        errs, runtime = fit_col(
            min_l_xy,
            hf,
            i,
            beta=args.beta,
            N_bc=args.N_bc,
            N_xy=args.N_pde_xy
        )
        L_inf_errs.append(errs[0])
        MSEs.append(errs[1])
        runtimes.append(runtime)
        if (i % 100) == 0:
            print(f"{i} / {N_problems}")
    
    results_path = Path(args.results_path) / uuid.uuid4().hex
    results_path.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({
        'L_inf_err': L_inf_errs,
        'MSE': MSEs,
        'runtime': runtimes
    })
    results.to_csv(results_path / 'results.csv')
    # Dump args to yaml
    with open(results_path / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f)


if __name__ == '__main__':
    main()
