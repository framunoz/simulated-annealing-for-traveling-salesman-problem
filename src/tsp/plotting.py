
import typing as t
from typing import Protocol, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .utils import _ArrayLike, _LiteralStyles

__all__ = ["plot_summary_sa", "plot_route", "SAStatsProtocol", "RouteProtocol"]


class RouteProtocol(Protocol):
    """Protocol for Route objects."""
    @property
    def route(self) -> list[int]:
        ...


class SAStatsProtocol(Protocol):
    """
    Protocol for Simulated Annealing statistics needed for plotting.
    Matches properties of SAStatsProxy.
    """
    @property
    def values(self) -> Sequence[float] | np.ndarray:
        ...
    
    @property
    def accept_prob(self) -> Sequence[float] | np.ndarray:
        ...
        
    @property
    def acceptance_ratio(self) -> Sequence[float] | np.ndarray:
        ...


def plot_summary_sa(model_sa: SAStatsProtocol, style: _LiteralStyles = "darkgrid"):
    """
    Plots a summary of the `model_sa` instance. More precisely, it plots the acceptance probabilities, acceptance
    ratios and values across iterations.

    :param model_sa: An object satisfying SAStatsProtocol (e.g. SAStatsProxy or SimulatedAnnealingTSP).
    :param style: The style of the plot. For further information see :func:`seaborn.axes_style`. Default is `darkgrid`.
    """
    with sns.axes_style(style):
        fig: Figure = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 1, hspace=0.2)
        ax1, ax2, ax3 = t.cast(tuple[Axes, Axes, Axes], gs.subplots(sharex=True))

        accept_prob = np.asarray(model_sa.accept_prob)
        # iterations = np.arange(len(accept_prob))

        ax1.scatter(
            np.arange(len(accept_prob)), accept_prob,
            1, marker="x"
        )
        ax1.set_ylabel("Probability")
        ax1.set_title("Acceptance probability across iterations")

        accept_ratio = np.asarray(model_sa.acceptance_ratio)

        ax2.plot(
            np.arange(len(accept_ratio)), accept_ratio
        )
        ax2.set_ylabel("Rate")
        ax2.set_title("Acceptance rate across iterations")

        values = np.asarray(model_sa.values)
        abs_min_values = np.minimum.accumulate(values)
        best_value = np.min(values)
        
        ax3.plot(np.arange(len(values)), values)
        ax3.plot(np.arange(len(abs_min_values)), abs_min_values)
        ax3.plot(np.arange(len(values)), np.ones_like(values) * best_value, "--")
        ax3.set_ylabel("Values")
        ax3.set_xlabel("Iterations")
        ax3.set_title("Route values across iterations")
        ax3.legend(["Values", "Best value at the moment", "Best value"])

    return fig, (ax1, ax2, ax3)


def plot_route(
        coords: _ArrayLike,
        route: RouteProtocol | None = None,
        lb: int | None = None,
        ub: int | None = None,
        style: _LiteralStyles = "darkgrid",
):
    """
    Plots the route given the coordinates.

    :param coords: The coordinates with shape ``(n_coords, n_dim)``
    :param route: Route. It is assumed to be correlative.
    :param lb: Lower bound.
    :param ub: Upper bound.
    :param style: The style of the plot. For further information see :func:`seaborn.axes_style`
    """

    route_indices = route.route if route is not None else list(range(len(coords)))  # type: ignore

    coords = np.asarray(coords)
    # Take coordinates in order
    # route_indices + route_indices[:1] to close the loop
    path_indices = route_indices + route_indices[:1]
    coords_ = coords.take(path_indices, 0)
    
    x, y = coords_[:, 0], coords_[:, 1]
    point_labels = [str(i) for i in range(len(route_indices))]

    with sns.axes_style(style):
        fig, ax = plt.subplots()
        ax.plot(x, y, "--b", alpha=0.5)
        ax.scatter(x, y, 100, c="r", alpha=0.7)

        for i, lab in enumerate(point_labels):
            # coords corresponds to the original points, so we index by 'i' directly assuming route_indices contains indices 0..N-1
            # But the label should be placed at the coordinate of the i-th city in the *route*? 
            # Original code: 
            # coords_ contains the points in visitation order.
            # point_labels contains 0..N-1.
            # loop enumerates labels.
            # The label '0' should go to the 0-th city in the list of cities, or the 0-th step in the route?
            # Original code: `plt.text(coords[i, 0], coords[i, 1] + 0.5, lab` <- this uses original coords array indexed by i (0..N-1).
            # So it labels the cities by their index in the original list, not the visitation order.
            ax.text(coords[i, 0], coords[i, 1] + 0.5, lab,
                     horizontalalignment='center', size='medium', color='black', weight='semibold')

        if lb is not None and ub is not None:
            ax.set_xlim(left=lb, right=ub)
            ax.set_ylim(bottom=lb, top=ub)
        
        ax.set_xlabel(r"Axis $X$")
        ax.set_ylabel(r"Axis $Y$")

        ax.axis("equal")

    return fig, ax
    