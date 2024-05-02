import linpde_gp.domains as domains
import matplotlib.axis
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from probnum.randprocs import GaussianProcess
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from linpde_gp.randprocs._gaussian_process import ConditionalGaussianProcess

from .plot_function import plot_at_time


def plot_gp_at_time(
    gp: GaussianProcess,
    t: float,
    domain: domains.Box,
    *,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.axis.Axis = None,
    zlims: list[float] = None,
    color: str = None,
    fidelity: int = 40,
    with_uncertainty=False,
):
    if fig is None and ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Xs = domain.uniform_grid((1, fidelity, fidelity), inset=(t, 0, 0))
    X, Y = Xs[0][..., 1], Xs[0][..., 2]

    H = gp.mean(Xs)[0]
    if with_uncertainty:
        S = np.sqrt(gp.cov.linop(Xs).diagonal())
        S = np.reshape(S, Xs.shape[:-1])
        #S = gp.std(Xs)[0]

    ax.plot_surface(X, Y, H, alpha=1, rcount=fidelity, ccount=fidelity)
    if with_uncertainty:
        ax.scatter(
            X.flatten(),
            Y.flatten(),
            H.flatten(),
            c=(1.96 * S).flatten(),
            cmap="coolwarm",
            s=5,
            alpha=0.8,
            vmin=0,
            vmax=1.96,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if zlims is not None:
        ax.set_zlim(zlims[0], zlims[1])
    return fig, ax


def plot_gp(
    gp: GaussianProcess,
    domain: domains.Box,
    num_t: int = 3,
    *,
    fig: matplotlib.figure.Figure = None,
    axs: matplotlib.axis.Axis = None,
    zlims: list[float] = None,
    color: str = None,
    fidelity: int = 40,
    skip_t0: bool = False,
    with_uncertainty=False,
):
    if fig is None:
        fig, axs = plt.subplots(1, num_t, figsize=(4 * num_t, 3))
    if skip_t0:
        ts = np.linspace(domain[0][0], domain[0][1], num_t + 1)[1:]
    else:
        ts = np.linspace(domain[0][0], domain[0][1], num_t)
    for t, ax in zip(ts, axs):
        plot_gp_at_time(
            gp,
            t,
            domain,
            fig=fig,
            ax=ax,
            zlims=zlims,
            color=color,
            fidelity=fidelity,
            with_uncertainty=with_uncertainty,
        )
    return fig, axs


def animate_gp(
    gp: GaussianProcess,
    domain: domains.Box,
    *,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.axis.Axis = None,
    zlims: list[float] = None,
    color: str = None,
    fidelity: int = 40,
    duration: float = None,
    fps: int = 30,
    notebook=True,
    use_tqdm=True,
    t_unit="sec",
    with_uncertainty=False,
):
    true_duration = domain[0][1] - domain[0][0]
    if duration is None:
        duration = true_duration
    num_frames = int(duration * fps)
    interval = 1000 / fps

    if fig is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X = domain.uniform_grid((num_frames, fidelity, fidelity))
    H = gp.mean(X)

    if with_uncertainty:
        print("Computing uncertainty")
        S = np.sqrt(gp.cov.linop(X).diagonal())
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S = np.reshape(S, H.shape)
        if isinstance(gp, ConditionalGaussianProcess):
            prior_covs = np.sqrt(gp._prior.cov.linop(X).diagonal())
            vmax = 1.96 * np.max(prior_covs)
        else:
            vmax = 1.96 * np.max(S)
        #S = gp.std(X)

    if zlims is None:
        zlims = [H.min() - 0.1, H.max() + 0.1]

    def animate(i):
        ax.clear()
        ax.plot_surface(
            X[i][..., 1],
            X[i][..., 2],
            H[i],
            rcount=fidelity,
            ccount=fidelity,
            color=color,
        )
        if with_uncertainty:
            ax.scatter(
                X[i][..., 1].flatten(),
                X[i][..., 2].flatten(),
                H[i].flatten(),
                c=(1.96 * S[i]).flatten(),
                cmap="coolwarm",
                s=5,
                alpha=0.8,
                vmin=0,
                vmax=vmax,
            )
        ax.set_title(f"t = {true_duration * i / float(num_frames):.2f} {t_unit}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if zlims is not None:
            ax.set_zlim(zlims)

    frames = range(num_frames)
    if use_tqdm and notebook:
        frames = tqdm_notebook(frames)
    elif use_tqdm:
        frames = tqdm(frames)
    anim = FuncAnimation(fig, animate, frames=frames, interval=interval)

    return anim
