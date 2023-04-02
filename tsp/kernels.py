import abc
import copy
from typing import Protocol

import numpy as np
from numpy.random import Generator

from tsp.utils import _GeneratorLike, _Route


class KernelTSP(metaclass=abc.ABCMeta):
    """
    Protocol class used to generalise the behaviour of sampling new routes.
    It is used in the :py:class:`SimulatedAnnealingTSP`.
    """
    def __init__(self, seed: _GeneratorLike = None):
        self.rng: Generator = np.random.default_rng(seed)

    @abc.abstractmethod
    def sample(self, route: _Route) -> _Route:
        """
        Samples a new route, using the previous route. You can call the instance to obtain the same effect.

        :param list[int] route: Previous route.
        :return: Modified route.
        """
        ...

    def __call__(self, route: _Route) -> _Route:
        """Alias for :py:method:`sample`"""
        return self.sample(route)


class SwapKernelTSP(KernelTSP):
    r"""
    Kernel that performs a swap between two cities on a route.

    e.g.: For a route ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``, if ``i=3`` and ``j=7`` are selected as indices to swap,
    the new route would be ``[0, 1, 2, 7, 4, 5, 6, 3, 8, 9]``.
    """

    @staticmethod
    def _sample(route: _Route, i: int, j: int) -> _Route:
        """To have a deterministic sampling. Used for testing."""
        new_route = copy.copy(route)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def sample(self, route: _Route) -> _Route:
        i, j = self.rng.choice(len(route), size=2, replace=False)
        return self._sample(route, i, j)


class ReversionKernelTSP(KernelTSP):
    r"""
    Kernel that performs a reversion on a route.

    e.g.: For a route ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``, if ``i=3`` and ``j=7`` are selected as indices to reverse
    the route, the new route would be ``[0, 1, 2, 7, 6, 5, 4, 3, 8, 9]``.
    """
    @staticmethod
    def _sample(route: _Route, i: int, j: int) -> _Route:
        i, j = min(i, j), max(i, j)
        if i == 0:
            return route[j::-1] + route[j+1:]
        return route[:i] + route[j:i-1:-1] + route[j+1:]

    def sample(self, route: _Route) -> _Route:
        i, j = self.rng.choice(len(route), size=2, replace=False)
        return self._sample(route, i, j)


class InsertionKernelTSP(KernelTSP):
    r"""
    Kernel that performs an insertion on a route.

    e.g.: For a route ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``, if ``i=3`` and ``j=7`` are selected as indices to insert
    the city 3 into the place 7, the new route would be ``[0, 1, 2, 4, 5, 6, 7, 3, 8, 9]``.
    """
    @staticmethod
    def _sample(route: _Route, i, j) -> _Route:
        new_route = copy.copy(route)
        new_route.insert(j, new_route.pop(i))
        return new_route

    def sample(self, route: _Route) -> _Route:
        i, j = self.rng.choice(len(route), size=2, replace=False)
        return self._sample(route, i, j)


class MixingKernelTSP(KernelTSP):
    """
    It is a mixing of the kernels. For example, if we have the kernels :math:`K_1`, :math:`K_2` and :math:`K_3`
    with the weights :math:`{p_i}_{i=1^{3}}`, then the resulting new kernel would be
    :math:`K = p_1 K_1 + p_2 K_2 + p_3 K_3`.
    """
    def __init__(self, kernels: list[KernelTSP], p: list[float], seed=None):
        super().__init__(seed)
        # Array of probabilities
        self.p = np.asarray(p)
        self.p = self.p / np.sum(self.p)

        # Ordering the array of probabilities
        indices = []
        candidates = {i: self.p[i] for i in range(len(self.p))}
        while candidates:
            argmax_candidate = max(candidates, key=candidates.get)
            indices.append(argmax_candidate)
            del candidates[argmax_candidate]

        self.p: np.ndarray = self.p.take(indices)
        self.kernels: list[KernelTSP] = [kernels[i] for i in indices]

        # Auxiliary variables
        self._cumsum_p: np.ndarray = np.concatenate([[0.], np.cumsum(self.p)])
        self._indexes = np.arange(len(self.p))

    def _sample(self, route: _Route, u: float) -> _Route:
        array_boolean = (self._cumsum_p[:-1] <= u) & (u < self._cumsum_p[1:])
        i_kernel: int = int(self._indexes[array_boolean])
        return self.kernels[i_kernel].sample(route)

    def sample(self, route: _Route) -> _Route:
        u = self.rng.uniform()
        return self._sample(route, u)
