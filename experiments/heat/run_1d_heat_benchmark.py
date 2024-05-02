import linpde_gp
import numpy as np
import probnum as pn
from gp_fvm.utils.figure_manager import FigureManager
from gp_fvm.plot.dim1 import plot_function, compare_to_solution_gp
from gp_fvm.finite_volumes import get_grid_from_depth
from linpde_gp.benchmarking import SolutionErrorEstimator
from gp_fvm.benchmark import evaluate_params
import pandas as pd

import argparse

pn.config.default_solver_linpde_gp = linpde_gp.solvers.CholeskySolver(dense=True)


def get_prior(l_t, l_x, output_scale):
    return pn.randprocs.GaussianProcess(
        mean=linpde_gp.functions.Zero(input_shape=(2,)),
        cov=output_scale**2
        * linpde_gp.randprocs.covfuncs.TensorProduct(
            linpde_gp.randprocs.covfuncs.Matern((), nu=1.5, lengthscales=l_t),
            linpde_gp.randprocs.covfuncs.Matern((), nu=2.5, lengthscales=l_x),
        ),
    )


def condition_ic(prior, problem, N_ic):
    X_ic = problem.initial_domain.uniform_grid(N_ic, inset=1e-6)
    Y_ic = problem.initial_condition.values(X_ic[..., 1])

    return prior.condition_on_observations(Y_ic, X_ic)


def condition_bc(prior, problem, N_bc):
    u_ic_bc = prior
    for bc in problem.boundary_conditions:
        X_bc = bc.boundary.uniform_grid(N_bc)
        Y_bc = bc.values(X_bc)

        u_ic_bc = u_ic_bc.condition_on_observations(Y_bc, X=X_bc)
    return u_ic_bc


def condition_fv(prior, depth, problem, box_domain):
    fv_domains = get_grid_from_depth(box_domain, depth)
    fv = linpde_gp.linfunctls.FiniteVolumeFunctional(fv_domains, problem.pde.diffop)
    return prior.condition_on_observations(L=fv, Y=np.zeros(fv_domains.shape))


def full_solve_fv(depth, l_t, l_x, problem, box_domain, output_scale, N_ic, N_bc):
    u_prior = get_prior(l_t, l_x, output_scale)
    u_ic = condition_ic(u_prior, problem, N_ic)
    u_ic_bc = condition_bc(u_ic, problem, N_bc)
    if depth == -1:
        return u_ic_bc
    u_fv = condition_fv(u_ic_bc, depth, problem, box_domain)
    return u_fv


def condition_collocation(prior, depth, problem, box_domain):
    points = box_domain.uniform_grid((2**depth, 2**depth), centered=True)
    return prior.condition_on_observations(
        Y=np.zeros(points.shape[:-1]), X=points, L=problem.pde.diffop
    )


def full_solve_collocation(
    depth, l_t, l_x, problem, box_domain, output_scale, N_ic, N_bc
):
    u_prior = get_prior(l_t, l_x, output_scale)
    u_ic = condition_ic(u_prior, problem, N_ic)
    u_ic_bc = condition_bc(u_ic, problem, N_bc)
    if depth == -1:
        return u_ic_bc
    u_fv = condition_collocation(u_ic_bc, depth, problem, box_domain)
    return u_fv


# Arg parsing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N-samples", type=int, default=20)
    parser.add_argument("--N-ic", type=int, default=15)
    parser.add_argument("--N-bc", type=int, default=50)
    parser.add_argument("--x-start", type=float, default=-1.0)
    parser.add_argument("--x-end", type=float, default=1.0)
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.1)

    # Lengthscale grid parameters
    parser.add_argument("--l-grid-start-t", type=float, default=1.0)
    parser.add_argument("--l-grid-end-t", type=float, default=5.0)
    parser.add_argument("--l-grid-step-t", type=float, default=0.1)
    parser.add_argument("--l-grid-start-x", type=float, default=1.0)
    parser.add_argument("--l-grid-end-x", type=float, default=5.0)
    parser.add_argument("--l-grid-step-x", type=float, default=0.1)

    parser.add_argument("--output-scale", type=float, default=1.0)

    # Maximum depth
    parser.add_argument("--max-depth", type=int, default=6)

    parser.add_argument("--output-folder", type=str, required=True)

    args = parser.parse_args()

    print("Running benchmark...")
    print(f"Args: {args}")
    np.random.seed(235092863)

    coefficients = np.random.normal(
        np.array([1.0, 0.5, 0.25, 0.125]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        size=(args.N_samples, 4),
    )

    spatial_domain = linpde_gp.domains.asdomain([args.x_start, args.x_end])
    temporal_domain = linpde_gp.domains.asdomain([args.t_start, args.t_end])
    alpha = args.alpha

    box_domain = linpde_gp.domains.Box([temporal_domain, spatial_domain])

    def get_problem(coefficients):
        return linpde_gp.problems.pde.HeatEquationDirichletProblem(
            t0=temporal_domain[0],
            T=temporal_domain[1],
            spatial_domain=spatial_domain,
            alpha=alpha,
            initial_values=linpde_gp.functions.TruncatedSineSeries(
                spatial_domain,
                coefficients=coefficients,
            ),
        )

    problems = [get_problem(coefficients[i]) for i in range(args.N_samples)]

    grid_l_t = np.arange(args.l_grid_start_t, args.l_grid_end_t, args.l_grid_step_t)
    grid_l_x = np.arange(args.l_grid_start_x, args.l_grid_end_x, args.l_grid_step_x)

    lengthscales_fv_per_problem = []
    errors_fv_per_problem = []
    lengthscales_col_per_problem = []
    errors_col_per_problem = []
    levels = np.arange(-1, args.max_depth + 1)

    output_scale = args.output_scale
    N_ic = args.N_ic
    N_bc = args.N_bc
    for idx, problem in enumerate(problems):
        print(f"Problem {idx+1}/{args.N_samples}")

        error_estimator = SolutionErrorEstimator(
            problem.solution, box_domain, norm="Linf"
        )

        lengthscales_fv, errors_fv = evaluate_params(
            levels,
            [grid_l_t, grid_l_x],
            lambda *args: full_solve_fv(
                *args, problem, box_domain, output_scale, N_ic, N_bc
            ),
            error_estimator,
            use_tqdm=False,
        )
        lengthscales_fv_per_problem.append(lengthscales_fv)
        errors_fv_per_problem.append(errors_fv)

        lengthscales_col, errors_col = evaluate_params(
            levels,
            [grid_l_t, grid_l_x],
            lambda *args: full_solve_collocation(
                *args, problem, box_domain, output_scale, N_ic, N_bc
            ),
            error_estimator,
            use_tqdm=False,
        )
        lengthscales_col_per_problem.append(lengthscales_col)
        errors_col_per_problem.append(errors_col)

    # Make dataframe where each row is a problem and each column is a level of FV or collocation
    df_errs_fv = pd.DataFrame(errors_fv_per_problem, columns=levels)
    df_errs_col = pd.DataFrame(errors_col_per_problem, columns=levels)
    df_lengthscales_fv = pd.DataFrame(lengthscales_fv_per_problem, columns=levels)
    df_lengthscales_col = pd.DataFrame(lengthscales_col_per_problem, columns=levels)

    # Save to disk
    df_errs_fv.to_csv(f"{args.output_folder}/errs_fv.csv")
    df_errs_col.to_csv(f"{args.output_folder}/errs_col.csv")
    df_lengthscales_fv.to_csv(f"{args.output_folder}/lengthscales_fv.csv")
    df_lengthscales_col.to_csv(f"{args.output_folder}/lengthscales_col.csv")

    print("Done.")


if __name__ == "__main__":
    main()
