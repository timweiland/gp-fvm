import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import linpde_gp
from linpde_gp.randprocs.covfuncs import TensorProductGrid
from linpde_gp.linfuncops.diffops import LinearDifferentialOperator, PartialDerivativeCoefficients, MultiIndex
from linpde_gp.linfunctls import (
    _EvaluationFunctional,
)
from gp_fvm.finite_volumes import get_grid_from_resolution
from linpde_gp.linfunctls import FiniteVolumeFunctional
import probnum as pn

from timeit import default_timer as timer

def read_data(filename):
    return h5py.File(filename)

def get_problem(hf, idx):
    return hf.get('tensor')[idx]

def get_ts(hf):
    return np.array(hf.get('t-coordinate')[:-1]).astype(np.float64)

def get_xs(hf):
    return np.array(hf.get('x-coordinate')).astype(np.float64)

def animate_problem(problem, xs, ts):
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.plot(xs, problem[frame])
        ax.set_title(f't = {ts[frame]:.2f} sec')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('u(t, x)')

    true_duration = 2.01
    num_frames = problem.shape[0]

    fps = num_frames/true_duration
    interval = 1000/fps
    return FuncAnimation(fig, update, frames=len(problem), interval=interval)

def animate_gp_sol(gp, domain, fps=30):
    fig, ax = plt.subplots()

    X_eval = domain.uniform_grid((int(fps * domain.bounds[0][1]), 40))
    print("Mean computation...")
    means = gp.mean(X_eval)
    print("Stds computation...")
    stds = np.sqrt(gp.cov.linop(X_eval).diagonal())
    stds = stds.reshape(means.shape)
    print("Done")

    def update(frame):
        ax.clear()
        ax.plot(X_eval[frame, :, 1], means[frame])
        ax.fill_between(
            X_eval[frame, :, 1],
            means[frame] - 1.96 * stds[frame],
            means[frame] + 1.96 * stds[frame],
            alpha=0.2,
            color='cornflowerblue',
        )
        # ax.set_title(f't = {X_eval[frame, :, 0]:.2f} sec')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('u(t, x)')

    interval = 1000/fps
    return FuncAnimation(fig, update, frames=X_eval.shape[0], interval=interval)

def get_prior(l_t, l_x, output_scale):
    return pn.randprocs.GaussianProcess(
        mean=linpde_gp.functions.Zero(input_shape=(2,)),
        cov=output_scale**2
        * linpde_gp.randprocs.covfuncs.TensorProduct(
            linpde_gp.randprocs.covfuncs.Matern((), nu=1.5, lengthscales=l_t),
            linpde_gp.randprocs.covfuncs.Matern((), nu=1.5, lengthscales=l_x),
        ),
    )

def get_diffop(beta):
    return LinearDifferentialOperator(
        coefficients=PartialDerivativeCoefficients(
            {
                (): {MultiIndex((1, 0)): 1.0, MultiIndex((0, 1)): beta},
            },
            (2,),
            (),
        ),
        input_shapes=((2,), ()))

def periodic_boundary_functional(X_left, X_right):
    L_left = _EvaluationFunctional(
            input_domain_shape=(2,),
            input_codomain_shape=(),
            X=X_left,
        )
    L_right = _EvaluationFunctional(
            input_domain_shape=(2,),
            input_codomain_shape=(),
            X=X_right,
        )
    return L_left - L_right

def gp_sol_errors(gp, sol, ts, xs):
    """
        Returns a tuple of (Linf, MSE) errors
    """
    assert sol.shape == (ts.size, xs.size)
    X_eval = TensorProductGrid(ts, xs)
    gp_preds = gp.mean(X_eval)
    diff = gp_preds - sol
    return np.abs(diff).max(), (diff**2).mean()

def fit_fv(l_t, l_x, hf, problem_idx, beta, N_pde_t=80, N_pde_x=80, ic_stride=10, N_bc=50, output_scale=1.0, ):
    start_time = timer()
    u_prior = get_prior(l_t, l_x, output_scale)
    problem = get_problem(hf, problem_idx)
    ts = get_ts(hf)
    xs = get_xs(hf)
    X_ic = TensorProductGrid([0.0], xs[::ic_stride])
    Y_ic = problem[0][::ic_stride].astype(np.float64)
    Y_ic = Y_ic.reshape(X_ic.shape[:-1])
    u_ic = u_prior.condition_on_observations(Y_ic, X_ic)

    spatial_domain = linpde_gp.domains.asdomain([0.0, 1.0])
    temporal_domain = linpde_gp.domains.asdomain([0., 2.0])
    domain = linpde_gp.domains.Box([temporal_domain, spatial_domain])

    X_left = domain.uniform_grid((N_bc, 1))
    X_right = domain.uniform_grid((N_bc, 1), inset=(0, domain.bounds[1][1]))

    L_boundary = periodic_boundary_functional(X_left, X_right)
    Y_boundary = np.zeros(L_boundary.output_shape)

    u_ic_bc = u_ic.condition_on_observations(Y=Y_boundary, L=L_boundary)

    D = get_diffop(beta)

    domains = get_grid_from_resolution(domain, [N_pde_t, N_pde_x])
    fv = FiniteVolumeFunctional(domains, D)
    u_all = u_ic_bc.condition_on_observations(L=fv, Y=np.zeros(domains.shape))

    errs = gp_sol_errors(u_all, problem, ts, xs)
    time_taken = timer() - start_time
    return errs, time_taken

def fit_col(l_t, l_x, hf, problem_idx, beta, N_pde_t=80, N_pde_x=80, ic_stride=10, N_bc=50, output_scale=1.0, ):
    start_time = timer()
    u_prior = get_prior(l_t, l_x, output_scale)
    problem = get_problem(hf, problem_idx)
    ts = get_ts(hf)
    xs = get_xs(hf)
    X_ic = TensorProductGrid([0.0], xs[::ic_stride])
    Y_ic = problem[0][::ic_stride].astype(np.float64)
    Y_ic = Y_ic.reshape(X_ic.shape[:-1])
    u_ic = u_prior.condition_on_observations(Y_ic, X_ic)

    spatial_domain = linpde_gp.domains.asdomain([0.0, 1.0])
    temporal_domain = linpde_gp.domains.asdomain([0., 2.0])
    domain = linpde_gp.domains.Box([temporal_domain, spatial_domain])

    X_left = domain.uniform_grid((N_bc, 1))
    X_right = domain.uniform_grid((N_bc, 1), inset=(0, domain.bounds[1][1]))

    L_boundary = periodic_boundary_functional(X_left, X_right)
    Y_boundary = np.zeros(L_boundary.output_shape)

    u_ic_bc = u_ic.condition_on_observations(Y=Y_boundary, L=L_boundary)

    D = get_diffop(beta)
    X_pde = domain.uniform_grid((N_pde_t, N_pde_x), centered=True)

    u_all = u_ic_bc.condition_on_observations(L=D, X=X_pde, Y=np.zeros(X_pde.shape[:-1]))

    errs = gp_sol_errors(u_all, problem, ts, xs)
    time_taken = timer() - start_time
    return errs, time_taken
