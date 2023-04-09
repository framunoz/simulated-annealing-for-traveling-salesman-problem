import abc
import copy
from typing import Protocol, Optional

import numpy as np
from numpy.random import Generator

from tsp.utils import _GeneratorLike, _Route, _EPS

__all__ = [
    "KernelTSP",
    "BaseKernelTSP",
    "SwapKernelTSP",
    "ReversionKernelTSP",
    "InsertionKernelTSP",
    "RandomWalkKernelTSP",
    "MixingKernelTSP",
]

_sample_doc = """
        Samples a new route, using the previous route. You can call the instance to obtain the same effect.
    
        :param route: Previous route.
        :return: Modified route.
        """

_call_doc = """Alias for :py:method:`sample`"""


class KernelTSP(Protocol):
    """
    Protocol class used to generalise the behaviour of sampling new routes.
    It is used in the :py:class:`tsp.sa.SimulatedAnnealingTSP`.
    """
    rng: Generator
    """The generator of the instance."""

    def sample(self, route: _Route) -> _Route:
        ...

    def __call__(self, route: _Route) -> _Route:
        ...


class BaseKernelTSP(metaclass=abc.ABCMeta):
    """
    Abstract class used to generalise the behaviour of sampling new routes.
    It is used in the :py:class:`tsp.sa.SimulatedAnnealingTSP`.
    """
    rng: Generator
    """The generator of the instance."""

    def __init__(self, seed: _GeneratorLike = None):
        self.rng: Generator = np.random.default_rng(seed)

    @abc.abstractmethod
    def sample(self, route: _Route) -> _Route:
        ...

    def __call__(self, route: _Route) -> _Route:
        return self.sample(route)


# Set the similar documentation
for _cls in [KernelTSP, BaseKernelTSP]:
    _cls.sample.__doc__ = _sample_doc
    _cls.__call__.__doc__ = _call_doc


class SwapKernelTSP(BaseKernelTSP):
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
        i, j = self.rng.choice(len(route)-1, size=2, replace=False) + 1
        return self._sample(route, i, j)


class ReversionKernelTSP(BaseKernelTSP):
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
        i, j = self.rng.choice(len(route)-1, size=2, replace=False) + 1
        return self._sample(route, i, j)


class InsertionKernelTSP(BaseKernelTSP):
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
        i, j = self.rng.choice(len(route)-1, size=2, replace=False) + 1
        return self._sample(route, i, j)


class RandomWalkKernelTSP(BaseKernelTSP):
    """
    Kernel that performs a random walk over possible path combinations. Returns a different path than the one
    initially given.
    """

    def sample(self, route: _Route) -> _Route:
        len_route = len(route)
        new_route = route
        while new_route == route:
            new_route = route[:1] + list(self.rng.choice(len_route-1, size=len_route-1, replace=False) + 1)
        return new_route


class MixingKernelTSP(BaseKernelTSP):
    r"""
    It is a mixing of the kernels. For example, if we have the kernels :math:`K_1`, :math:`K_2` and :math:`K_3`
    with the weights :math:`\{p_i\}_{i=1}^{3}`, then the resulting new kernel would be
    :math:`K = p_1 K_1 + p_2 K_2 + p_3 K_3`.
    """
    def __init__(
            self,
            kernels: list[KernelTSP] | list[tuple[float, KernelTSP]],
            probs: Optional[list[float]] = None,
            seed: _GeneratorLike = None,
    ):
        super().__init__(seed)
        self._len_kernels = len(kernels)

        if probs is not None and len(probs) != len(kernels):
            raise ValueError("The lengths of 'kernels' and 'probs' must be equals.")

        # Array of probabilities
        if probs is None:
            probs = [p for p, k in kernels]
            kernels = [k for p, k in kernels]
        probs = np.asarray(probs) + _EPS * len(probs)
        probs = probs / np.sum(probs)

        # Ordering the array of probabilities
        indices = []
        candidates = {i: probs[i] for i in range(len(probs))}
        while candidates:
            argmax_candidate = max(candidates, key=candidates.get)
            indices.append(argmax_candidate)
            del candidates[argmax_candidate]

        self.probs: np.ndarray = probs.take(indices)
        self.kernels: list[KernelTSP] = [kernels[i] for i in indices]

        # Auxiliary variables
        self._cumsum_p: np.ndarray = np.concatenate([[0.], np.cumsum(self.probs)])

    def _choose_kernel(self, u: float) -> KernelTSP:
        """Choose a kernel given a number between 0 and 1."""
        # Case u = 1 ==> Return the last kernel
        if u == 1.:
            return self.kernels[-1]
        kernel = None
        for i in range(self._len_kernels):
            lb, ub = self._cumsum_p[i], self._cumsum_p[i+1]
            if lb <= u < ub:
                kernel = self.kernels[i]
                break
        return kernel

    def _sample(self, route: _Route, u: float) -> _Route:
        kernel = self._choose_kernel(u)
        return kernel.sample(route)

    def sample(self, route: _Route) -> _Route:
        u = self.rng.uniform()
        return self._sample(route, u)
