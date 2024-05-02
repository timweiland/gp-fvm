import linpde_gp
import numpy as np
import probnum as pn
from gp_fvm.utils.figure_manager import FigureManager
from gp_fvm.plot.dim1 import plot_function, compare_to_solution_gp
from gp_fvm.finite_volumes import get_grid_from_depth
from linpde_gp.benchmarking import SolutionErrorEstimator
from linpde_gp.linfuncops.diffops import TimeDerivative
from gp_fvm.benchmark import evaluate_params
import pandas as pd

import argparse

pn.config.default_solver_linpde_gp = linpde_gp.solvers.CholeskySolver(dense=True)


def get_prior(l_t, l_x, l_y, output_scale):
    return pn.randprocs.GaussianProcess(
        mean=linpde_gp.functions.Zero(input_shape=(3,)),
        cov=output_scale**2
        * linpde_gp.randprocs.covfuncs.TensorProduct(
            linpde_gp.randprocs.covfuncs.Matern((), nu=2.5, lengthscales=l_t),
            linpde_gp.randprocs.covfuncs.Matern((), nu=2.5, lengthscales=l_x),
            linpde_gp.randprocs.covfuncs.Matern((), nu=2.5, lengthscales=l_y),
        ),
    )


def condition_ic(prior, problem, N_ic_xy):
    X_ic = problem.initial_domain.uniform_grid((N_ic_xy, N_ic_xy))
    Y_ic = problem.initial_condition.values(X_ic[..., 1:])

    u_ic = prior.condition_on_observations(X=X_ic, Y=Y_ic)
    return u_ic.condition_on_observations(X=X_ic, Y=Y_ic, L=TimeDerivative((3,)))


def condition_bc(prior, box_domain, N_bc_t, N_bc_spatial):
    boundary_x = box_domain.uniform_grid((N_bc_t, N_bc_spatial, 2))
    boundary_y = box_domain.uniform_grid((N_bc_t, 2, N_bc_spatial))

    u_bc = prior.condition_on_observations(X=boundary_x, Y=np.zeros(boundary_x.shape[:-1]))
    return u_bc.condition_on_observations(X=boundary_y, Y=np.zeros(boundary_y.shape[:-1]))


def condition_fv(prior, depth, problem, box_domain):
    domains = get_grid_from_depth(box_domain, depth)
    fv = linpde_gp.linfunctls.FiniteVolumeFunctional(domains, problem.pde.diffop)

    if depth <= 4:
        solver = linpde_gp.solvers.CholeskySolver(dense=True)
    else:
        solver = linpde_gp.solvers.itergp.IterGP_CG_Solver(threshold=1e-2, max_iterations=5000, num_actions_compressed=5000, eval_points=None, loggers=[])
    return prior.condition_on_observations(L=fv, Y=np.zeros(domains.shape), solver=solver, fresh_start=depth > 4)


def full_solve_fv(depth, l_t, l_xy, problem, box_domain, output_scale, N_ic_xy, N_bc_t, N_bc_spatial):
    u_prior = get_prior(l_t, l_xy, l_xy, output_scale)
    u_ic = condition_ic(u_prior, problem, N_ic_xy)
    u_ic_bc = condition_bc(u_ic, box_domain, N_bc_t, N_bc_spatial)
    if depth == -1:
        return u_ic_bc
    u_fv = condition_fv(u_ic_bc, depth, problem, box_domain)
    return u_fv


def condition_collocation(prior, depth, problem, box_domain):
    points = box_domain.uniform_grid((2 ** depth, 2 ** depth, 2 ** depth), centered=True)

    if depth <= 4:
        solver = linpde_gp.solvers.CholeskySolver(dense=True)
    else:
        solver = linpde_gp.solvers.itergp.IterGP_CG_Solver(threshold=1e-2, max_iterations=5000, num_actions_compressed=5000, eval_points=None, loggers=[])
    return prior.condition_on_observations(Y=np.zeros(points.shape[:-1]), X=points, L=problem.pde.diffop, solver=solver, fresh_start=depth > 4)


def full_solve_collocation(
    depth, l_t, l_xy, problem, box_domain, output_scale, N_ic_xy, N_bc_t, N_bc_spatial
):
    u_prior = get_prior(l_t, l_xy, l_xy, output_scale)
    u_ic = condition_ic(u_prior, problem, N_ic_xy)
    u_ic_bc = condition_bc(u_ic, box_domain, N_bc_t, N_bc_spatial)
    if depth == -1:
        return u_ic_bc
    u_fv = condition_collocation(u_ic_bc, depth, problem, box_domain)
    return u_fv


# Arg parsing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N-samples", type=int, default=20)
    parser.add_argument("--N-ic-xy", type=int, default=8)
    parser.add_argument("--N-bc-t", type=int, default=20)
    parser.add_argument("--N-bc-spatial", type=int, default=10)
    parser.add_argument("--x-end", type=float, default=1.0)
    parser.add_argument("--y-end", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--c", type=float, default=1.0)

    # Lengthscale grid parameters
    parser.add_argument("--l-grid-start-t", type=float, default=0.1)
    parser.add_argument("--l-grid-end-t", type=float, default=2.0)
    parser.add_argument("--N-grid-t", type=int, default=4)
    parser.add_argument("--l-grid-start-xy", type=float, default=0.1)
    parser.add_argument("--l-grid-end-xy", type=float, default=1.0)
    parser.add_argument("--N-grid-xy", type=int, default=4)

    parser.add_argument("--output-scale", type=float, default=1.0)

    # Maximum depth
    parser.add_argument("--max-depth", type=int, default=5)

    parser.add_argument("--output-folder", type=str, required=True)

    args = parser.parse_args()

    print("Running benchmark...")
    print(f"Args: {args}")
    np.random.seed(235092863)

    mean = np.array([
        [1.0, -0.5],
        [-0.5, 0.],
        ])
    cov = np.array([
        [0.1, 0.2],
        [0.2, 0.3],
        ])
    coefficients = np.random.normal(mean, cov, size=(args.N_samples,) + mean.shape)

    box_domain = linpde_gp.domains.Box([[0, args.duration], [0, args.x_end], [0, args.y_end]])
    spatial_domain = linpde_gp.domains.Box([[0., args.x_end], [0., args.y_end]])

    def get_problem(coefficients):
        return linpde_gp.problems.pde.WaveEquationDirichletProblem(
            t0=0.,
            T=args.duration,
            spatial_domain=spatial_domain,
            c=args.c,
            initial_values=linpde_gp.functions.TruncatedSineSeries(
                spatial_domain,
                coefficients=coefficients,
            ),
        )

    problems = [get_problem(coefficients[i]) for i in range(args.N_samples)]

    grid_l_t = np.linspace(args.l_grid_start_t, args.l_grid_end_t, args.N_grid_t)
    grid_l_xy = np.linspace(args.l_grid_start_xy, args.l_grid_end_xy, args.N_grid_xy)

    lengthscales_fv_per_problem = []
    errors_fv_per_problem = []
    lengthscales_col_per_problem = []
    errors_col_per_problem = []
    levels = np.arange(-1, args.max_depth + 1)

    output_scale = args.output_scale
    N_ic_xy = args.N_ic_xy
    N_bc_t = args.N_bc_t
    N_bc_spatial = args.N_bc_spatial
    for idx, problem in enumerate(problems):
        print(f"Problem {idx+1}/{args.N_samples}")

        error_estimator = SolutionErrorEstimator(
            problem.solution, box_domain, norm="Linf"
        )

        lengthscales_fv, errors_fv = evaluate_params(
            levels,
            [grid_l_t, grid_l_xy],
            lambda *args: full_solve_fv(
                *args, problem, box_domain, output_scale, N_ic_xy, N_bc_t, N_bc_spatial
            ),
            error_estimator,
            use_tqdm=False,
        )
        lengthscales_fv_per_problem.append(lengthscales_fv)
        errors_fv_per_problem.append(errors_fv)

        lengthscales_col, errors_col = evaluate_params(
            levels,
            [grid_l_t, grid_l_xy],
            lambda *args: full_solve_collocation(
                *args, problem, box_domain, output_scale, N_ic_xy, N_bc_t, N_bc_spatial
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
