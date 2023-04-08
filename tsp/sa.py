from collections import deque
from typing import Optional

import numpy as np
from numpy.random import Generator

from tsp.kernels import SwapKernelTSP, KernelTSP
from tsp.utils import _Route, _GeneratorLike, _EPS, exponential_cooling_schedule, _ArrayLike


class SimulatedAnnealingTSP:
    """
    The Simulated Annealing algorithm for solving the Traveling Salesman Problem.

    The novelty of this class, is that you can choose the kernel (which is equivalent to choosing the
    connections in the graph) from the list of kernels in the module :py:mod:`tsp.kernels`.
    """
    def __init__(
            self,
            dist_matrix: _ArrayLike,
            n_iter: int,
            cooling_schedule=exponential_cooling_schedule,
            kernel: Optional[KernelTSP] = None,
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

        # For statistics
        self._log_accept_prob: list[float] = []
        self._n_accept: int = 0
        self._acception_ratio: list[float] = []
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
        return np.array(self._values)

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

        route_to_return = list(self.rng.choice(self.len_route, size=self.len_route, replace=False))

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

        for k in range(self._last_k, self.n_iter + self._last_k):
            # Sample a new route from the kernel
            new_route, dist_new_route = self._register_route(self.kernel(route))

            # Compute the acceptance probability
            gap = dist_new_route - dist_route
            log_accept_prob = 0 if gap <= 0 else -gap / (self.temp(k) + _EPS)
            self._log_accept_prob.append(log_accept_prob)

            if log_accept_prob >= 0 or self.rng.uniform() <= np.exp(log_accept_prob):
                # Update the best route
                self._update_best_route(route, dist_new_route)
                self._n_accept += 1

                # Set the new route as the route
                route, dist_route = new_route, dist_new_route

            self._acception_ratio.append(self._n_accept / (k + 1))
            self._last_k = k + 1

        self._last_route = route
