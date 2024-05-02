import linpde_gp.domains as domains
import matplotlib.axis
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt


def plot_at_time(
    f: callable,
    t: float,
    domain: domains.Box,
    *,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.axis.Axis = None,
    ylims: list[float] = None,
    label: str = None,
    color: str = None,
    fidelity: int = 100,
):
    if fig is None:
        fig, ax = plt.subplots()
    Xs = domain.uniform_grid((1, fidelity), inset=(t, 0))
    ax.plot(Xs[0][:, 1], f(Xs)[0], label=label, color=color)
    ax.set_title(f"t = {t:.2f}")
    if ylims is not None:
        ax.set_ylim(ylims)


def plot_function(
    f: callable,
    domain: domains.Box,
    num_t: int = 3,
    *,
    fig: matplotlib.figure.Figure = None,
    axs: matplotlib.axis.Axis = None,
    ylims: list[float] = None,
    label: str = None,
    color: str = None,
    fidelity: int = 100,
    skip_t0: bool = False,
):
    # Plot at num_t evenly spaced times
    if fig is None:
        fig, axs = plt.subplots(1, num_t, figsize=(4 * num_t, 3))
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
            ylims=ylims,
            label=label,
            color=color,
            fidelity=fidelity,
        )
    return fig, axs
