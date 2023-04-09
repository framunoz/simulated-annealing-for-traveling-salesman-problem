from collections import deque
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.random import Generator

from tsp.kernels import SwapKernelTSP, KernelTSP
from tsp.utils import _Route, _GeneratorLike, _EPS, exponential_cooling_schedule, cooling_schedule_function, _ArrayLike, \
    _LiteralStyles

__all__ = [
    "SimulatedAnnealingTSP",
    "plot_summary_sa",
]


class SimulatedAnnealingTSP:
    """
    The Simulated Annealing algorithm for solving the Traveling Salesman Problem.

    The novelty of this class, is that you can choose the kernel (which is equivalent to choosing the
    connections in the graph) from the list of kernels in the module :py:mod:`tsp.kernels`.
    """
    def __init__(
            self,
            dist_matrix: _ArrayLike,
            n_iter: int = 1_000_000,
            cooling_schedule: cooling_schedule_function = exponential_cooling_schedule(),
            kernel: Optional[KernelTSP] = None,
            early_stop: bool = True,
            stop_after: int = 10_000,
            seed: _GeneratorLike = None
    ):
        """
        Class initialiser.

        :param dist_matrix: Matrix of distances. Must be a two-dimensional square matrix.
        :param n_iter: The number of iterations to run the algorithm.
        :param cooling_schedule: Cooling schedule function. Must be a function that receives an integer
        :math:`k` and returns the temperature.
        :param kernel: The kernel of the algorithm. This can be any of the :py:mod:`tsp.kernels`.
        The default is :py:class:`SwapKernelTSP`.
        :param early_stop: In case it is desired to stop iterations before reaching the maximum number of iterations.
        default is True.
        :param stop_after: The number of iterations to stop iterations, if no other route has been accepted.
        :param seed: Seed or generator. The kernel generator will be used if this parameter is None.
        """
        # Define the distance matrix and some asserts.
        self.dist_matrix: np.ndarray = np.asarray(dist_matrix)
        if len(self.dist_matrix.shape) != 2:
            raise ValueError("The distance matrix must have dimension 2,"
                             f" not {len(self.dist_matrix.shape)}.")
        n, m = self.dist_matrix.shape
        if n != m:
            raise ValueError(f"The distance matrix must be square, not {n, m}.")

        # Normalise the distance matrix for stability
        self.dist_matrix /= np.max(self.dist_matrix)

        # Define the length of the route.
        self.len_route = n

        self.n_iter: int = n_iter
        self.temp = cooling_schedule

        # Set the kernel. Default is SwapKernelTSP.
        self.kernel: KernelTSP = kernel or SwapKernelTSP(seed)

        # Set the generator. Default is the kernel generator.
        self.rng: Generator = np.random.default_rng(seed)
        if seed is None:
            self.rng = self.kernel.rng

        self._early_stop = early_stop
        self._stop_after = stop_after

        # For statistics
        self._log_accept_prob: list[float] = []
        self._n_accept: int = 0
        self._acceptance_ratio: list[float] = []
        self._values: list[float] = []
        self._best_route: dict[str, float | _Route | None] = {
            "value": float("inf"),
            "route": None,
        }
        self._last_route = None
        self._last_k = 0

    @property
    def values(self) -> np.ndarray:
        """The values of the total distances of the routes."""
        return np.array(self._values[1:])

    @property
    def best_route(self) -> _Route:
        """The best route found by the algorithm."""
        return self._best_route["route"]

    @property
    def best_value(self) -> float:
        """The best value found by the algorithm."""
        return self._best_route["value"]

    @property
    def accept_prob(self) -> np.ndarray:
        """The acceptance probability history."""
        return np.exp(np.array(self._log_accept_prob))

    @property
    def accept_ratio(self) -> np.ndarray:
        return np.array(self._acceptance_ratio)

    @property
    def abs_min_values(self) -> np.ndarray:
        """The absolute minimum value."""
        return np.minimum.accumulate(self.values)

    def dist_route(self, route: _Route) -> float:
        r"""
        Compute the total distance the traveller should travel when executing the route.
        If the route is of the form :math:`x = (x_0, \ldots, x_n)`, then the total distance is computed
        by

        .. math::
            \sum_{k=0}^{n} d(x_k, x_{k+1})

        With the convention that :math:`x_{n+1} = x_0`.

        :param route: The route to compute the total distance.
        :return: The total distance.
        """
        rot_route = deque(route)
        rot_route.rotate(-1)
        total_dist = sum([self.dist_matrix[i, j] for i, j in zip(route, rot_route)])
        return total_dist

    def _init_route(self):
        """Compute the initial route."""
        if self._last_route is not None:
            return self._last_route

        route_to_return = list(range(self.len_route))

        return route_to_return

    def _register_route(self, route: _Route):
        """Records route properties for later statistical purposes."""
        dist_route = self.dist_route(route)
        self._values.append(dist_route)
        return route, dist_route

    def _update_best_route(self, route: _Route, dist_route: float):
        """Updates the best route, in case there was one."""
        if dist_route < self._best_route["value"]:
            self._best_route.update(value=dist_route, route=route)

    def run(self):
        """Execute the algorithm."""
        # Initial route
        route, dist_route = self._register_route(self._init_route())
        self._update_best_route(route, dist_route)

        stop_counter = 0
        for k in range(self._last_k, self.n_iter + self._last_k):
            stop_counter += 1

            # Sample a new route from the kernel
            new_route, dist_new_route = self._register_route(self.kernel(route))

            # Compute the acceptance probability
            gap = dist_new_route - dist_route
            log_accept_prob = 0 if gap <= 0 else -gap / (self.temp(k) + _EPS)
            self._log_accept_prob.append(log_accept_prob)

            if log_accept_prob >= 0 or self.rng.uniform() <= np.exp(log_accept_prob):
                stop_counter = 0
                # Update the best route
                self._update_best_route(route, dist_new_route)
                self._n_accept += 1

                # Set the new route as the route
                route, dist_route = new_route, dist_new_route

            self._acceptance_ratio.append(self._n_accept / (k + 1))
            self._last_k = k + 1

            if self._early_stop and stop_counter >= self._stop_after:
                break

        self._last_route = route


def plot_summary_sa(model_sa: SimulatedAnnealingTSP, style: _LiteralStyles = "darkgrid"):
    with sns.axes_style(style):
        fig: mpl.figure.Figure = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 1, hspace=0.2)
        ax1, ax2, ax3 = gs.subplots(sharex=True)

        ax1: mpl.axes.Axes
        ax2: mpl.axes.Axes
        ax3: mpl.axes.Axes

        iterations = np.arange(len(model_sa.accept_prob))

        ax1.scatter(
            iterations, model_sa.accept_prob,
            1, marker="x"
        )
        ax1.set_ylabel("Probability")
        ax1.set_title("Acceptance probability across iterations")

        ax2.plot(
            iterations, model_sa.accept_ratio
        )
        ax2.set_ylabel("Rate")
        ax2.set_title("Acceptance rate across iterations")

        ax3.plot(iterations, model_sa.values)
        ax3.plot(iterations, model_sa.abs_min_values)
        ax3.plot(iterations, np.ones_like(model_sa.values) * model_sa.best_value, "--")
        ax3.set_ylabel("Values")
        ax3.set_xlabel("Iterations")
        ax3.set_title("Route values across iterations")
        ax3.legend(["Values", "Best value at the moment", "Best value"])

