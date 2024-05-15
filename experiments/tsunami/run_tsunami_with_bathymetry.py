import argparse
from pathlib import Path

import linpde_gp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import probnum as pn
import torch
import rasterio
import pickle
from linpde_gp.linfuncops import SelectOutput
from linpde_gp.linfuncops.diffops import (
    MultiIndex,
    PartialDerivativeCoefficients,
    get_shallow_water_diffops_2D,
)
from linpde_gp.randprocs.covfuncs import (
    IndependentMultiOutputCovarianceFunction,
    Matern,
    TensorProduct,
)
from linpde_gp.solvers.itergp import IterGPSolver
from linpde_gp.solvers.itergp.loggers import FileLogger, PickleLogger, PlotLogger
from linpde_gp.solvers.itergp.policies import CGPolicy, PredefinedPolicy, SwitchPolicy, VarianceCGPolicy
from linpde_gp.solvers.itergp.stopping_criteria import (
    IterationStoppingCriterion,
    ResidualNormStoppingCriterion,
)
from gp_fvm.finite_volumes import get_grid_from_resolution
from tueplots import bundles
from timeit import default_timer as timer

pn.config.default_solver_linpde_gp = linpde_gp.solvers.CholeskySolver(dense=True)


def gaussian_bump_2d(X, mean, cov, scaling_factor=1.0):
    """Compute a 2D Gaussian bump at the given mean and covariance."""
    inv = np.linalg.inv(cov)
    X_tilde = X - mean
    return scaling_factor * np.exp(
        -0.5 * (X_tilde[..., None, :] @ (inv @ X_tilde[..., None]))[..., 0, 0]
    )


def get_tensor_product(lengthscales):
    return TensorProduct(
        Matern((), nu=2.5, lengthscales=lengthscales[0]),
        Matern((), nu=2.5, lengthscales=lengthscales[1]),
        Matern((), nu=2.5, lengthscales=lengthscales[2]),
    )


prior_mean = linpde_gp.functions.StackedFunction(
    (
        linpde_gp.functions.Constant((3,), 0.0),
        linpde_gp.functions.Constant((3,), 0.0),
        linpde_gp.functions.Constant((3,), 0.0),
    )
)


def get_prior(lengthscales):
    prior_cov = IndependentMultiOutputCovarianceFunction(
        get_tensor_product(lengthscales),
        get_tensor_product(lengthscales),
        get_tensor_product(lengthscales),
    )
    return pn.randprocs.GaussianProcess(mean=prior_mean, cov=prior_cov)


def condition_on_open_boundary(
    u: pn.randprocs.GaussianProcess,
    domain: linpde_gp.domains.Box,
    c_fn: pn.functions.Function,
    N_bc_t: int,
    N_bc_x: int,
    N_bc_y: int,
):
    # Define boundary differential operators
    B1_coeffs = PartialDerivativeCoefficients(
        {(0,): {MultiIndex([1, 0, 0]): 1.0, MultiIndex([0, 1, 0]): c_fn}}, (3,), (3,)
    )
    D_B1 = linpde_gp.linfuncops.diffops.LinearDifferentialOperator(
        B1_coeffs, ((3,), (3,))
    )

    B3_coeffs = PartialDerivativeCoefficients(
        {(0,): {MultiIndex([1, 0, 0]): 1.0, MultiIndex([0, 1, 0]): -c_fn}}, (3,), (3,)
    )
    D_B3 = linpde_gp.linfuncops.diffops.LinearDifferentialOperator(
        B3_coeffs, ((3,), (3,))
    )

    B2_coeffs = PartialDerivativeCoefficients(
        {(0,): {MultiIndex([1, 0, 0]): 1.0, MultiIndex([0, 0, 1]): -c_fn}}, (3,), (3,)
    )
    D_B2 = linpde_gp.linfuncops.diffops.LinearDifferentialOperator(
        B2_coeffs, ((3,), (3,))
    )

    B4_coeffs = PartialDerivativeCoefficients(
        {(0,): {MultiIndex([1, 0, 0]): 1.0, MultiIndex([0, 0, 1]): c_fn}}, (3,), (3,)
    )
    D_B4 = linpde_gp.linfuncops.diffops.LinearDifferentialOperator(
        B4_coeffs, ((3,), (3,))
    )

    # Condition on boundary observations
    X_B1 = domain.uniform_grid((N_bc_t, 1, N_bc_y), inset=(0, domain.bounds[1][1], 0))
    X_B2 = domain.uniform_grid((N_bc_t, N_bc_x, 1))
    X_B3 = domain.uniform_grid((N_bc_t, 1, N_bc_y))
    X_B4 = domain.uniform_grid((N_bc_t, N_bc_x, 1), inset=(0, 0, domain.bounds[2][1]))
    Y_boundary_B1_B3 = np.zeros((N_bc_t, 1, N_bc_y))
    Y_boundary_B2_B4 = np.zeros((N_bc_t, N_bc_x, 1))

    u_ic_bc = u
    u_ic_bc = u_ic_bc.condition_on_observations(
        X=X_B1, Y=Y_boundary_B1_B3, L=D_B1, noise=1e-8
    )
    u_ic_bc = u_ic_bc.condition_on_observations(
        X=X_B3, Y=Y_boundary_B1_B3, L=D_B3, noise=1e-8
    )
    u_ic_bc = u_ic_bc.condition_on_observations(
        X=X_B2, Y=Y_boundary_B2_B4, L=D_B2, noise=1e-8
    )
    u_ic_bc = u_ic_bc.condition_on_observations(
        X=X_B4, Y=Y_boundary_B2_B4, L=D_B4, noise=1e-8
    )
    return u_ic_bc


def condition_on_closed_boundary(
    u: pn.randprocs.GaussianProcess,
    domain: linpde_gp.domains.Box,
    N_bc_t: int,
    N_bc_x: int,
    N_bc_y: int,
):
    X_boundary_x = domain.uniform_grid((N_bc_t, 2, N_bc_y))
    X_boundary_y = domain.uniform_grid((N_bc_t, N_bc_x, 2))

    Y_boundary_x = np.zeros((N_bc_t, 2, N_bc_y))
    Y_boundary_y = np.zeros((N_bc_t, N_bc_x, 2))

    u_ic_bc = u.condition_on_observations(
        X=X_boundary_x, Y=Y_boundary_x, L=SelectOutput(((3,), (3,)), 1)
    )
    u_ic_bc = u_ic_bc.condition_on_observations(
        X=X_boundary_y, Y=Y_boundary_y, L=SelectOutput(((3,), (3,)), 2)
    )

    from linpde_gp.linfuncops.diffops import MultiIndex, PartialDerivativeCoefficients

    dx_coeffs = PartialDerivativeCoefficients(
        {(0,): {MultiIndex([0, 1, 0]): 1.0}}, (3,), (3,)
    )
    dx = linpde_gp.linfuncops.diffops.LinearDifferentialOperator(
        dx_coeffs, ((3,), (3,))
    )
    dy_coeffs = PartialDerivativeCoefficients(
        {(0,): {MultiIndex([0, 0, 1]): 1.0}}, (3,), (3,)
    )
    dy = linpde_gp.linfuncops.diffops.LinearDifferentialOperator(
        dy_coeffs, ((3,), (3,))
    )

    u_ic_bc = u_ic_bc.condition_on_observations(X=X_boundary_x, Y=Y_boundary_x, L=dx)
    return u_ic_bc.condition_on_observations(X=X_boundary_y, Y=Y_boundary_y, L=dy)


def generate_partitioned_arrays(original_shape, dividing_shape):
    # Calculate the size of each block in each dimension
    block_sizes = tuple(o // d for o, d in zip(original_shape, dividing_shape))

    # Generate the partitioned arrays using np.ndindex
    arrays = []
    for indices in np.ndindex(*dividing_shape):
        # Create an array filled with 1's
        arr = np.zeros(original_shape, dtype=np.float64)

        slices = []
        for i, b, end in zip(indices, block_sizes, dividing_shape):
            if i == end - 1:
                slices.append(slice(i * b, None))
            else:
                slices.append(slice(i * b, (i + 1) * b))
        arr[tuple(slices)] = 1

        arrays.append(arr)

    return arrays


def condition_fvm(
    u: pn.randprocs.GaussianProcess,
    H_fn: pn.functions.Function,
    coriolis: float,
    domain: linpde_gp.domains.Box,
    N_vols_t: int,
    N_vols_x: int,
    N_vols_y: int,
    N_precondition_t: int,
    N_precondition_x: int,
    N_precondition_y: int,
    iterations: int,
    iterations_uncertainty_reduction: int,
    logfile: str,
    output_folder: Path,
):
    g = 9.81 / 1000  # km/s^2

    SW1, SW2, SW3 = get_shallow_water_diffops_2D(H_fn, g, coriolis)
    domains_fine = get_grid_from_resolution(
        domain, [N_vols_t, N_vols_x, N_vols_y]
    )
    print(f"Domains shape: {domains_fine.shape}")
    fv_fine_1 = linpde_gp.linfunctls.FiniteVolumeFunctional(domains_fine, SW1)
    fv_fine_2 = linpde_gp.linfunctls.FiniteVolumeFunctional(domains_fine, SW2)
    fv_fine_3 = linpde_gp.linfunctls.FiniteVolumeFunctional(domains_fine, SW3)
    Y_fine = np.zeros(domains_fine.shape)
    precondition_vecs_small = generate_partitioned_arrays(
        (N_vols_t, N_vols_x, N_vols_y), (N_precondition_t, N_precondition_x, N_precondition_y)
    )
    prior_size = u._gram_matrix.shape[1]
    added_size = 3 * N_vols_t * N_vols_x * N_vols_y
    total_size = prior_size + added_size
    precondition_vecs = []
    for vec in precondition_vecs_small:
        precondition_vec = np.zeros(total_size)
        # Repeat flattened vec three times
        precondition_vec[prior_size:] = np.tile(vec.flatten(), 3)
        precondition_vec /= added_size
        precondition_vecs.append(precondition_vec)
    precondition_vecs = np.array(precondition_vecs)
    precondition_vecs = torch.from_numpy(precondition_vecs).to("cuda" if torch.cuda.is_available() else "cpu")

    eval_points = domain.uniform_grid((10, 5, 5), bounds=((1200, 1800), (40, 75), (33.3, 66.6)))
    print(f"Eval points: {eval_points}")

    picklefile = output_folder / "itergp_data.pkl"
    plot_logger = PlotLogger()
    file_logger = FileLogger(logfile)
    pickle_logger = PickleLogger(picklefile)
    loggers = [file_logger, pickle_logger, plot_logger]

    # policy = SwitchPolicy(
    #     PredefinedPolicy(precondition_vecs),
    #     CGPolicy(),
    #     int(precondition_vecs.shape[0]),
    # )

    policy = SwitchPolicy(
        CGPolicy(),
        VarianceCGPolicy(eval_points),
        iterations,
    )

    # policy = CGPolicy()

    N_iterations_total = iterations + iterations_uncertainty_reduction
    solver = IterGPSolver(
        policy=policy,
        stopping_criterion=IterationStoppingCriterion(N_iterations_total)
        | ResidualNormStoppingCriterion(1e-2),
        eval_points=None,
        num_actions_compressed=N_iterations_total,
        loggers=loggers,
    )

    u_sol = u
    u_sol = u_sol.condition_on_observations(
        L=fv_fine_1, Y=Y_fine, noise=1e-8, solver=solver, fresh_start=True
    )
    u_sol = u_sol.condition_on_observations(
        L=fv_fine_2, Y=Y_fine, noise=1e-8, solver=solver
    )
    u_sol = u_sol.condition_on_observations(
        L=fv_fine_3, Y=Y_fine, noise=1e-8, solver=solver
    )
    u_sol.solver.store_K_hat_inverse_approx = False
    return u_sol, plot_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iterations", type=int, default=300)
    parser.add_argument("--num-iterations-uncertainty-reduction", type=int, default=100)
    parser.add_argument(
        "--bathymetry-path",
        type=str,
        required=True,
    )
    parser.add_argument("--duration", type=int, default=30 * 60)  # seconds
    parser.add_argument("--ic-spacing-km", type=float, default=3)
    parser.add_argument("--bc-spacing-km", type=float, default=5)
    parser.add_argument("--bc-spacing-sec", type=float, default=30)
    parser.add_argument("--volume-length-km-x", type=float, default=1)
    parser.add_argument("--volume-length-km-y", type=float, default=5)
    parser.add_argument("--volume-length-sec", type=float, default=20)
    parser.add_argument("--coriolis", type=float, default=0.0)
    parser.add_argument(
        "--logfile",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ffmpeg-path",
        type=str,
        default=None,
    )
    # Trigger open or closed boundary
    parser.add_argument(
        "--open-boundary",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    print("Arguments:")
    print(args)

    # Dump arguments to file
    with open(Path(args.output_folder) / "args.txt", "w") as f:
        f.write(str(args))

    if args.ffmpeg_path is not None:
        plt.rcParams["animation.ffmpeg_path"] = args.ffmpeg_path

    # Open the GeoTIFF file
    with rasterio.open(args.bathymetry_path) as src:
        # Read the raster data into a NumPy array
        raster_array = src.read(1)
        bounds = src.bounds
        # Get coordinate array
        cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
        lons, lats = rasterio.transform.xy(src.transform, rows, cols)
        lons = np.array(lons)
        lats = np.array(lats)

    lon_start = bounds.left
    lon_end = bounds.right
    lat_start = bounds.bottom
    lat_end = bounds.top

    raster_array_flipped = raster_array[::-1, :]

    km_per_lat = 111.32
    km_per_lon = km_per_lat * np.cos(np.radians((lat_start + lat_end) / 2))
    print(f"km per lat: {km_per_lat:.2f}km, km per lon: {km_per_lon:.2f}km")

    dist_x = (lons[-1, -1] - lon_start) * km_per_lon
    dist_y = (lat_end - lats[-1, -1]) * km_per_lat
    print(f"dist_x: {dist_x:.2f}km, dist_y: {dist_y:.2f}km")

    domain = linpde_gp.domains.Box(
        [[0.0, args.duration], [0.0, dist_x], [0.0, dist_y]]
    )  # time, x, y

    earthquake_epicenter = (142.516, 38.062)  # (longitude, latitude)
    earthquake_epicenter_coords_km = (
        earthquake_epicenter[0] - lon_start
    ) * km_per_lon, (earthquake_epicenter[1] - lat_start) * km_per_lat
    print(
        f"earthquake epicenter coordinates (km): {earthquake_epicenter_coords_km[0]:.2f}km, {earthquake_epicenter_coords_km[1]:.2f}km"
    )

    mean = np.array(earthquake_epicenter_coords_km)
    cov = np.array([[20, 0], [0, 300]])
    scaling_factor = 5

    ic_spacing_km = args.ic_spacing_km # One initial condition observation every 3km
    N_ic_x = int(dist_x / ic_spacing_km)
    N_ic_y = int(dist_y / ic_spacing_km)
    print(f"Initial conditions: {N_ic_x} x {N_ic_y} = {N_ic_x * N_ic_y}")
    X_ic = domain.uniform_grid((1, N_ic_x, N_ic_y))
    Y_ic = np.zeros((1, N_ic_x, N_ic_y, 3))
    Y_ic[..., 0] = gaussian_bump_2d(
        X_ic[..., 1:], mean, cov, scaling_factor=scaling_factor
    )

    lengthscale_t = 60
    lengthscale_x = 6
    lengthscale_y = 10

    lengthscales = np.array([lengthscale_t, lengthscale_x, lengthscale_y])
    u_prior = get_prior(lengthscales)

    print("Conditioning on initial conditions...")
    u_ic = u_prior.condition_on_observations(X=X_ic, Y=Y_ic, noise=1e-8)

    kms_lon = (lons - lon_start) * km_per_lon
    kms_lat = (lats - lat_start) * km_per_lat

    def closest_idx(x: np.ndarray):
        dists = np.sqrt((kms_lon - x[0]) ** 2 + (kms_lat - x[1]) ** 2)
        dists_flat = dists.flatten()
        flat_idx = np.argmin(dists_flat)
        return np.unravel_index(flat_idx, dists.shape)

    def bathymetry_vals(X: np.ndarray):
        X_flat = X.reshape(-1, 2)
        idxs = np.array([closest_idx(x) for x in X_flat])
        return raster_array_flipped[idxs[..., 0], idxs[..., 1]].reshape(X.shape[:-1])

    def propagation_speeds(X: np.ndarray):
        baths = bathymetry_vals(X[..., 1:])
        abs_speeds = np.sqrt(9.81 * np.abs(baths)) / 1000
        return -np.sign(baths) * abs_speeds

    c_fn = pn.functions.LambdaFunction(propagation_speeds, (3,), ())

    def H_vals(X: np.ndarray):
        if X.ndim == 4:
            t0_vals = -bathymetry_vals(X[0][..., 1:]) / 1000
            # Repeat that for all time steps
            t0_vals = t0_vals.reshape((1, *t0_vals.shape))
            return np.tile(t0_vals, (X.shape[0], 1, 1))
        return -bathymetry_vals(X[..., 1:]) / 1000

    H_fn = pn.functions.LambdaFunction(H_vals, (3,), ())
    # H_fn = 1.0

    bc_spacing_km = args.bc_spacing_km  # One boundary observation every 5km
    N_bc_x = int(dist_x / bc_spacing_km)
    N_bc_y = int(dist_y / bc_spacing_km)

    bc_spacing_sec = args.bc_spacing_sec  # One boundary observation every 30 seconds
    N_bc_t = int(args.duration / bc_spacing_sec)
    print(
        f"Boundary observations: {N_bc_t} x {N_bc_x} x {N_bc_y} = {N_bc_t * N_bc_x * N_bc_y}"
    )

    print("Conditioning on boundary...")
    if args.open_boundary:
        u_ic_bc = condition_on_open_boundary(u_ic, domain, c_fn, N_bc_t, N_bc_x, N_bc_y)
    else:
        u_ic_bc = condition_on_closed_boundary(u_ic, domain, N_bc_t, N_bc_x, N_bc_y)

    print("Conditioning on FVM...")

    N_volume_x = int(dist_x / args.volume_length_km_x)
    N_volume_y = int(dist_y / args.volume_length_km_y)
    N_volume_t = int(args.duration / args.volume_length_sec)
    print(
        f"Volumes: {N_volume_t} x {N_volume_x} x {N_volume_y} = {N_volume_t * N_volume_x * N_volume_y}"
    )

    output_folder = Path(args.output_folder)
    u_sol, plot_logger = condition_fvm(
        u_ic_bc,
        H_fn,
        args.coriolis,
        domain,
        N_volume_t,
        N_volume_x,
        N_volume_y,
        4,
        6,
        5,
        args.num_iterations,
        args.num_iterations_uncertainty_reduction,
        args.logfile,
        output_folder,
    )
    print("Computing representer weights...")
    print(f"Gram matrix shape: {u_sol._gram_matrix.shape}")
    start_time = timer()
    u_sol.representer_weights
    end_time = timer()
    print(f"Time: {end_time - start_time:.2f}s")

    print("Plotting animation...")

    def plot_at_fixed_time(X, mean, fig=None, ax=None):
        # bath_vals = bathymetry_vals(X[..., 1:])
        t = X[0, 0, 0]
        if fig is None or ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
        # ax.plot_surface(X[..., 1], X[..., 2], bath_vals, cmap=cmocean.cm.topo, norm=divnorm)

        ax.plot_surface(
            X[..., 1], X[..., 2], mean[..., 0], alpha=1.0, color="cornflowerblue"
        )
        ax.set_zlim(-5, 5)
        # # Rotate viewpoint
        # ax.view_init(azim=45)
        ax.set_box_aspect([dist_x / dist_y, 1, 1])
        ax.set_title(f"t = {t/60:.1f} mins")

    FPS = 20

    def animated_plot_batched(X, means):
        fps = FPS
        interval = 1000 / fps
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        def animate(i):
            ax.clear()
            plot_at_fixed_time(X[i], means[i], fig=fig, ax=ax)

        anim = FuncAnimation(
            fig, animate, frames=range(means.shape[0]), interval=interval
        )
        return anim

    X = domain.uniform_grid((120, 65, int(65 * dist_y / dist_x)))
    print("Computing means...")
    means = u_sol.mean(X)
    print("Computing covs...")
    covs = u_sol.cov.linop(X).diagonal()
    Hs = H_fn(X)
    print("Saving...")
    gp_eval = {
        "X": X,
        "means": means,
        "covs": covs,
        "Hs": Hs,
    }
    with open(output_folder / "gp_eval.pkl", "wb") as f:
        pickle.dump(gp_eval, f)

    print("Animating...")
    anim = animated_plot_batched(X, means)

    FFwriter = FFMpegWriter(fps=FPS, extra_args=["-vcodec", "libx264"])
    anim.save(output_folder / "gp_mean.mp4", writer=FFwriter)

    print("Plotting IterGP errors...")
    plt.rcParams.update(bundles.icml2022())

    print("Saving action matrix...")
    action_mat = u_sol.solver.action_matrix
    for block in action_mat.blocks[0, :]:
        print(block.shape)
    with open(output_folder / "action_matrix.pkl", "wb") as f:
        pickle.dump(action_mat.blocks[0, 1].todense(), f)

    print("Saving slice matrix...")
    X_slice = domain.uniform_grid((120, 50, 1), inset=(0, 0, 50))
    Hs_slice = H_fn(X_slice)
    means_slice = u_sol.mean(X_slice)
    print("Computing diagonal...")
    covs_slice = u_sol.cov.linop(X_slice).diagonal()
    print("Computing dense matrix...")
    cov_mat_slice_linop = u_sol.cov.linop(X_slice)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cov_mat_slice_dense = torch.zeros(cov_mat_slice_linop.shape, dtype=torch.float64, device=device)
    N = cov_mat_slice_dense.shape[1]
    print_points = [int(N * i / 10) for i in range(10)]
    for i in range(cov_mat_slice_dense.shape[1]):
        if i in print_points:
            print(f"Reached {i} / {N}")
        x = torch.zeros(cov_mat_slice_linop.shape[1], dtype=torch.float64, device=device)
        x[i] = 1
        cov_mat_slice_dense[:, i] = cov_mat_slice_linop @ x

    slice_data = {
        "X": X_slice,
        "Hs": Hs_slice,
        "means": means_slice,
        "covs": covs_slice,
        "cov_mat": cov_mat_slice_dense.cpu().numpy(),
    }
    with open(output_folder / "slice_data.pkl", "wb") as f:
        pickle.dump(slice_data, f)

    err_img_path = output_folder / "rel_errs.pdf"
    fig, _ = plot_logger.plot_error()
    fig.savefig(err_img_path)

    print("Done.")


if __name__ == "__main__":
    main()
