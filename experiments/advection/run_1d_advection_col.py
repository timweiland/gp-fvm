import argparse
import linpde_gp
import numpy as np
import probnum as pn
from gp_fvm.utils.figure_manager import FigureManager
from gp_fvm.plot.dim1 import plot_function, compare_to_solution_gp, plot_gp

from advection_utils import read_data, get_problem, get_ts, get_xs, animate_problem, fit_col
from matplotlib import pyplot as plt

import pandas as pd

from pathlib import Path
import uuid

import optuna
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf', type=str, default='../data/1D_Advection_Sols_beta0.4.hdf5')
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--ic-stride', type=int, default=10)
    parser.add_argument('--N-bc', type=int, default=50)
    parser.add_argument('--N-pde-t', type=int, default=50)
    parser.add_argument('--N-pde-x', type=int, default=100)
    parser.add_argument('--results-path', type=str, default="./results")
    args = parser.parse_args()
    print(args)

    hf = read_data(args.hdf)

    N_problems_search = 10

    def hyperopt_objective(trial):
        l_t = trial.suggest_float('l_t', 1e-4, 3.0)
        l_x = trial.suggest_float('l_x', 1e-4, 3.0)
        mean_MSE = 0.0
        for i in range(N_problems_search):
            errs, _ = fit_col(
                l_t,
                l_x,
                hf, 
                i, 
                beta=args.beta, 
                N_pde_t=args.N_pde_t, 
                N_pde_x=args.N_pde_x, 
                ic_stride=args.ic_stride, 
                N_bc = args.N_bc)
            mean_MSE += errs[1]
        mean_MSE /= N_problems_search
        return mean_MSE

    print("Optimizing hyperparameters...")
    study = optuna.create_study(sampler=optuna.samplers.GPSampler())
    study.optimize(hyperopt_objective, n_trials=100)
    print(f"Best trial: {study.best_trial}")
    min_l_t = study.best_params['l_t']
    min_l_x = study.best_params['l_x']

    N_problems = hf.get('tensor').shape[0]
    L_inf_errs = []
    MSEs = []
    runtimes = []
    for i in range(N_problems):
        errs, runtime = fit_col(
            min_l_t, 
            min_l_x, 
            hf, 
            i, 
            beta=args.beta, 
            N_pde_t=args.N_pde_t, 
            N_pde_x=args.N_pde_x, 
            ic_stride=args.ic_stride, 
            N_bc = args.N_bc)
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
