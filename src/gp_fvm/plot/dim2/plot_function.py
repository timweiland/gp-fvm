import linpde_gp.domains as domains
import matplotlib.axis
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.animation import FuncAnimation

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


def plot_at_time(
    f: callable,
    t: float,
    domain: domains.Box,
    *,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.axis.Axis = None,
    zlims: list[float] = None,
    color: str = None,
    fidelity: int = 40,
):
    if fig is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Xs = domain.uniform_grid((1, fidelity, fidelity), inset=(t, 0, 0))

    ax.plot_surface(
        Xs[0][..., 1],
        Xs[0][..., 2],
        f(Xs)[0],
        rcount=fidelity,
        ccount=fidelity,
        color=color,
    )
    ax.set_title(f"t = {t:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if zlims is not None:
        ax.set_zlim(zlims)

def plot_at_time_heatmap(
    f: callable,
    t: float,
    domain: domains.Box,
    *,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.axis.Axis = None,
    fidelity: int = 40,
    vmin: float = None,
    vmax: float = None,
):
    if fig is None:
        fig, ax = plt.subplots()
    Xs = domain.uniform_grid((1, fidelity, fidelity), inset=(t, 0, 0))

    ax.imshow(
        f(Xs)[0].transpose(),
        origin="lower",
        extent=[domain[2][0], domain[2][1], domain[1][0], domain[1][1]],
        vmin=vmin,
        vmax=vmax,
        cmap="seismic",
    )
    ax.set_title(f"t = {t:.2f} sec")
    # Only two ticks per axis
    ax.set_xticks([domain[2][0], domain[2][1]])
    ax.set_yticks([domain[1][0], domain[1][1]])
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")

def get_min_max(
    f: callable,
    ts: list[float],
    domain: domains.Box,
    fidelity: int = 40,
):
    min = np.inf
    max = -np.inf
    for t in ts:
        Xs = domain.uniform_grid((1, fidelity, fidelity), inset=(t, 0, 0))
        f_vals = f(Xs)[0]
        if f_vals.min() < min:
            min = f_vals.min()
        if f_vals.max() > max:
            max = f_vals.max()
    return min, max

def plot_heatmap(
    f: callable,
    domain: domains.Box,
    n_rowcol = 2,
    *,
    fig: matplotlib.figure.Figure = None,
    axs: matplotlib.axis.Axis = None,
    fidelity: int = 40,
    skip_t0: bool = False,
):
    # Plot at num_t evenly spaced times
    if fig is None:
        # n_rowcol x n_rowcol grid of plots with spacing
        fig, axs = plt.subplots(
            n_rowcol,
            n_rowcol,
            layout="compressed"
        )
    num_t = n_rowcol * n_rowcol
    if skip_t0:
        ts = np.linspace(domain[0][0], domain[0][1], num_t + 1)[1:]
    else:
        ts = np.linspace(domain[0][0], domain[0][1], num_t)
    axs_flat = [ax for row in axs for ax in row]

    vmin, vmax = get_min_max(f, ts, domain, fidelity)
    for t, ax in zip(ts, axs_flat):
        plot_at_time_heatmap(
            f,
            t,
            domain,
            fig=fig,
            ax=ax,
            fidelity=fidelity,
            vmin=vmin,
            vmax=vmax,
        )

    # Add global axis labels
    fig.text(0.46, 0.02, "x", ha="center")
    fig.text(0.15, 0.5, "y", va="center", rotation="vertical")
    # Add colorbar
    fig.colorbar(axs_flat[0].get_images()[0], ax=axs_flat, shrink=0.6)

    # Remove ticks from all but outer plots
    for i in range(n_rowcol):
        for j in range(n_rowcol):
            if i != n_rowcol - 1:
                axs[i, j].set_xticks([])
            if j != 0:
                axs[i, j].set_yticks([])

    return fig, axs


def plot_function(
    f: callable,
    domain: domains.Box,
    num_t: int = 3,
    *,
    fig: matplotlib.figure.Figure = None,
    axs: matplotlib.axis.Axis = None,
    zlims: list[float] = None,
    color: str = None,
    fidelity: int = 40,
    skip_t0: bool = False,
):
    # Plot at num_t evenly spaced times
    if fig is None:
        fig, axs = plt.subplots(1, num_t, subplot_kw={"projection": "3d"})
    if skip_t0:
        ts = np.linspace(domain[0][0], domain[0][1], num_t + 1)[1:]
    else:
        ts = np.linspace(domain[0][0], domain[0][1], num_t)
    for t, ax in zip(ts, axs):
        plot_at_time(
            f,
            t,
            domain,
            fig=fig,
            ax=ax,
            zlims=zlims,
            color=color,
            fidelity=fidelity,
        )
    return fig, axs


def animate_function(
    f: callable,
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
    t_unit = "sec"
):
    true_duration = domain[0][1] - domain[0][0]
    if duration is None:
        duration = true_duration
    num_frames = int(duration * fps)
    interval = 1000 / fps

    if fig is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X = domain.uniform_grid((num_frames, fidelity, fidelity))
    f_vals = f(X)

    if zlims is None:
        zlims = [f_vals.min() - 0.1, f_vals.max() + 0.1]

    def animate(i):
        ax.clear()
        ax.plot_surface(
            X[i][..., 1],
            X[i][..., 2],
            f_vals[i],
            rcount=fidelity,
            ccount=fidelity,
            color=color,
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
    anim = FuncAnimation(
        fig, animate, frames=frames, interval=interval
    )

    return anim

def show_anim_notebook(anim: FuncAnimation):
    from IPython.display import HTML

    return HTML(anim.to_jshtml())

def animate_sample(
    X_vals: np.ndarray,
    sample_vals: np.ndarray,
    ts: np.ndarray,
    duration: float,
    zlims = None,
):
    num_frames = ts.size
    fps = num_frames / duration
    interval = 1000 / fps

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    if zlims is None:
        zlims = [sample_vals.min() - 0.1, sample_vals.max() + 0.1]

    def animate(i):
        ax.clear()
        ax.plot_surface(
            X_vals[i][..., 1],
            X_vals[i][..., 2],
            sample_vals[i],
            rcount=X_vals.shape[1],
            ccount=X_vals.shape[2],
        )
        ax.set_title(f"t = {ts[i]:.2f} sec")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if zlims is not None:
            ax.set_zlim(zlims)

    frames = range(num_frames)
    anim = FuncAnimation(
        fig, animate, frames=frames, interval=interval
    )

    return anim