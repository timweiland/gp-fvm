import h5py
import numpy as np
import probnum as pn
import linpde_gp
from linpde_gp.linfuncops.diffops import LinearDifferentialOperator, PartialDerivativeCoefficients, MultiIndex
from gp_fvm.finite_volumes import get_grid_from_resolution
from linpde_gp.linfunctls import FiniteVolumeFunctional

from timeit import default_timer as timer

def read_data(filename):
    return h5py.File(filename)

def get_nu_fn(hf, problem_idx):
    xs = np.array(hf.get('x-coordinate')).astype(np.float64)
    ys = np.array(hf.get('y-coordinate')).astype(np.float64)
    nu_discrete = hf.get('nu')[problem_idx]

    @np.vectorize
    def nu(x, y):
        x_idx = np.abs(x - xs).argmin()
        y_idx = np.abs(y - ys).argmin()
        return nu_discrete[x_idx, y_idx]

    def nu_vec(X):
        if X.ndim == 1:
            return nu(X[0], X[1])
        return nu(X[..., 0], X[..., 1])

    return pn.functions.LambdaFunction(nu_vec, (2,))

def get_diffop(nu_fn):
    return LinearDifferentialOperator(
        coefficients=PartialDerivativeCoefficients(
            {
                (): {MultiIndex((2, 0)): nu_fn, MultiIndex((0, 2)): nu_fn},
            },
            (2,),
            (),
        ),
        input_shapes=((2,), ()))

def get_prior(l_xy):
    return pn.randprocs.GaussianProcess(
        mean=linpde_gp.functions.Zero(input_shape=(2,)),
        cov=linpde_gp.randprocs.covfuncs.TensorProduct(
            linpde_gp.randprocs.covfuncs.Matern((), nu=2.5, lengthscales=l_xy),
            linpde_gp.randprocs.covfuncs.Matern((), nu=2.5, lengthscales=l_xy),
        ),
    )

xy_domain = linpde_gp.domains.asdomain([0.0, 1.0])
domain = linpde_gp.domains.Box([xy_domain, xy_domain])

def errors(gp, hf, problem_idx):
    X_eval = domain.uniform_grid((128, 128))
    preds = gp.mean(X_eval)
    diff = preds - hf.get('tensor')[problem_idx][0]
    MSE = (diff**2).mean()
    max_error = np.abs(diff).max()
    return max_error, MSE

def fit_fv(l_xy, hf, problem_idx, beta, N_bc=30, N_xy=50):
    start_time = timer()
    u_prior = get_prior(l_xy)
    X_bc1 = domain.uniform_grid((2, N_bc))
    X_bc2 = domain.uniform_grid((N_bc, 2))
    Y_bc1 = np.zeros(X_bc1.shape[:-1])
    Y_bc2 = np.zeros(X_bc2.shape[:-1])

    u_bc1 = u_prior.condition_on_observations(X=X_bc1, Y=Y_bc1)
    u_bc = u_bc1.condition_on_observations(X=X_bc2, Y=Y_bc2)

    nu_fn = get_nu_fn(hf, problem_idx)
    D = get_diffop(nu_fn)
    domains = get_grid_from_resolution(domain, [N_xy, N_xy])
    fv = FiniteVolumeFunctional(domains, D)
    time_taken = timer() - start_time
    u_all = u_bc.condition_on_observations(L=fv, Y=(-beta) * (1/N_xy**2) * np.ones(domains.shape))
    return errors(u_all, hf, problem_idx), time_taken

def fit_col(l_xy, hf, problem_idx, beta, N_bc=30, N_xy=50):
    start_time = timer()
    u_prior = get_prior(l_xy)
    X_bc1 = domain.uniform_grid((2, N_bc))
    X_bc2 = domain.uniform_grid((N_bc, 2))
    Y_bc1 = np.zeros(X_bc1.shape[:-1])
    Y_bc2 = np.zeros(X_bc2.shape[:-1])

    u_bc1 = u_prior.condition_on_observations(X=X_bc1, Y=Y_bc1)
    u_bc = u_bc1.condition_on_observations(X=X_bc2, Y=Y_bc2)

    nu_fn = get_nu_fn(hf, problem_idx)
    D = get_diffop(nu_fn)

    X_pde = domain.uniform_grid((N_xy, N_xy), inset=(1e-4, 1e-4))
    u_all = u_bc.condition_on_observations(L=D, X=X_pde, Y=(-beta) * np.ones(X_pde.shape[:-1]))
    time_taken = timer() - start_time
    return errors(u_all, hf, problem_idx), time_taken

