import linpde_gp.domains as domains
import matplotlib.axis
import matplotlib.figure
from probnum.randprocs import GaussianProcess

from .plot_function import plot_function
from .plot_gp import plot_gp


def compare_to_solution_gp(
    gp: GaussianProcess,
    solution: callable,
    domain: domains.Box,
    num_t: int = 3,
    *,
    fig: matplotlib.figure.Figure = None,
    axs: matplotlib.axis.Axis = None,
    label: str = None,
    ylims: list[float] = None,
    fidelity: int = 100,
    skip_t0: bool = False,
):
    fig, axs = plot_gp(
        gp,
        domain,
        num_t=num_t,
        fig=fig,
        axs=axs,
        ylims=ylims,
        label=label,
        color="cornflowerblue",
        fidelity=fidelity,
        skip_t0=skip_t0,
    )
    fig, axs = plot_function(
        solution,
        domain,
        fig=fig,
        axs=axs,
        ylims=ylims,
        num_t=num_t,
        label="Ground truth",
        color="gold",
        fidelity=fidelity,
        skip_t0=skip_t0,
    )
    for ax in axs:
        ax.legend()
    return fig, axs
