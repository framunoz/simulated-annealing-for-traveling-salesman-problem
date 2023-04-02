from collections import deque

import numpy as np
from numpy._typing import ArrayLike
from numpy.random import Generator

from tsp.kernels import KernelTSP, SwapKernelTSP
from tsp.utils import _Route, _GeneratorLike

_EPS = np.finfo("float64").eps


# noinspection PyPep8Naming
def default_cooling_schedule(k, T_0=25, k_0=1) -> float:
    r"""
    The default cooling schedule is

    .. math::
        T_k = (T_0 \cdot \ln(k_0) \ln(k + 1 + k_0))^{-1}

    :param int k: The number of the iteration at the temperature :math:`T_k`.
    :param float T_0: Initial temperature. Default is 25
    :param float k_0: Iteration phase shift. Default is 1
    :return: The temperature at iteration :math:`k`.
    """
    return (T_0 * np.log(k_0)) / (np.log(k + k_0))


class SimulatedAnnealingTSP:
    """
    The Simulated Annealing algorithm for solving the Traveling Salesman Problem.

    The novelty of this class, is that you can choose the kernel (which is equivalent to choosing the connections in the graph)
    from the list of kernels in the module :py:module:`tsp.kernels`.
    """
    def __init__(
            self,
            dist_matrix: ArrayLike,
            n_iter: int,
            cooling_schedule=default_cooling_schedule,
            kernel: KernelTSP = None,
            seed: _GeneratorLike = None
    ):
        """
        Class initialiser.

        :param dist_matrix: Matrix of distances. Must be a two-dimensional square matrix.
        :param n_iter: The number of iterations to run the algorithm.
        :param cooling_schedule: Cooling schedule function. Must be a function that receives an integer :math:`k` and returns the temperature.
        :param kernel: The kernel of the algorithm. This can be any of the :py:module:`tsp.kernels`. The default is :py:class:`SwapKernelTSP`.
        :param seed: Seed or generator. The kernel generator will be used if this parameter is None.
        """
        # Define the distance matrix and some asserts.
        self.dist_matrix: np.ndarray = np.asarray(dist_matrix)
        if len(self.dist_matrix.shape) != 2:
            raise ValueError(f"The distance matrix must have dimension 2, not {len(self.dist_matrix.shape)}.")
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
        self._accept_prob: list[float] = []
        self._values: list[float] = []
        self._best_route: dict[str, float | _Route | None] = {
            "value": None,
            "route": None,
        }

    @property
    def accept_prob(self):
        """The acceptance probability history."""
        return np.array(self._accept_prob)

    @property
    def values(self):
        """The values of the total distances of the routes."""
        return np.array(self._values)

    @property
    def abs_min_values(self):
        """The absolute minimum value."""
        return np.minimum.accumulate(self.values)

    @property
    def best_route(self):
        """The best route found by the algorithm."""
        return np.array(self._best_route["route"])

    def dist_route(self, route: _Route) -> float:
        r"""Compute the total distance the traveller should travel when executing the route.
        If the route is of the form :math:`x = (x_0, \ldots, x_n)`, then the total distance is computed by means of

        .. math::
            \sum_{k = 0}^{n} d(x_k, x_{k+1})

        With the convention that :math:`x_{n+1} = x_0`.

        :param route: The route to compute the total distance.
        :return: The total distance.
        """
        rot_route = deque(route)
        rot_route.rotate(-1)
        total_dist = sum([self.dist_matrix[i, j] for i, j in zip(route, rot_route)])
        return total_dist

    def _init_route(self):
        """Compute a greedy route to facilitate optimisation."""
        # Start with the first city
        route_to_return = [0]
        # Set of next candidates
        candidates = {i for i in range(1, self.len_route)}
        # While there are candidates...
        while candidates:
            last_city = route_to_return[-1]
            # Obtain the distance between the last city and the candidates.
            dist_candidates = {k: self.dist_matrix[last_city, k] for k in candidates}
            # Get the candidate city with less distance
            argmin_candidate = min(dist_candidates, key=dist_candidates.get)
            route_to_return.append(argmin_candidate)
            candidates.remove(argmin_candidate)

        return route_to_return

    def run(self):
        """Execute the algorithm."""
        # Initial route
        route = self._init_route()
        dist_route = self.dist_route(route)
        self._values.append(dist_route)
        self._best_route.update(value=dist_route, route=route)

        for k in range(self.n_iter):
            # Sample a new route from the kernel
            new_route = self.kernel.sample(route)
            dist_new_route = self.dist_route(new_route)
            self._values.append(dist_new_route)

            # Compute the acceptance probability
            gap = dist_new_route - dist_route
            accept_prob = np.exp(0 if gap <= 0 else -gap / (self.temp(k) + _EPS))
            self._accept_prob.append(accept_prob)

            if dist_new_route <= 0 or self.rng.uniform() <= accept_prob:
                # Update the best route
                if self._best_route["value"] >= dist_new_route:
                    self._best_route.update(value=dist_new_route, route=new_route)

                # Set the new route as the route
                route, dist_route = new_route, dist_new_route

