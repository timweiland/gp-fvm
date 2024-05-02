import linpde_gp.domains as domains
import matplotlib.axis
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt

from .plot_function import plot_at_time

from probnum.randprocs import GaussianProcess


def plot_gp(
    gp: GaussianProcess,
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
    mean, std = gp.mean, gp.std
    if fig is None:
        fig, axs = plt.subplots(1, num_t, figsize=(4 * num_t, 3))
    if skip_t0:
        ts = np.linspace(domain[0][0], domain[0][1], num_t + 1)[1:]
    else:
        ts = np.linspace(domain[0][0], domain[0][1], num_t)
    for t, ax in zip(ts, axs):
        plot_at_time(
            mean,
            t,
            domain,
            fig=fig,
            ax=ax,
            ylims=ylims,
            label=label,
            color=color,
            fidelity=fidelity,
        )
        Xs = domain.uniform_grid((1, fidelity), inset=(t, 0))
        ax.fill_between(
            Xs[0][:, 1],
            mean(Xs)[0] - 1.96 * std(Xs)[0],
            mean(Xs)[0] + 1.96 * std(Xs)[0],
            alpha=0.2,
            color=color,
        )
    return fig, axs
