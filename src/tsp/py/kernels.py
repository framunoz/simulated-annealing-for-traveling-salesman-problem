
import abc
import copy
import typing as t
from typing import Optional, Protocol, Sequence

import numpy as np
from numpy.random import Generator

from ..utils import _EPS, _GeneratorLike
from .elements import Route

__all__ = [
    "Kernel",
    "BaseKernel",
    "SwapKernel",
    "ReversionKernel",
    "InsertionKernel",
    "RandomWalkKernel",
    "MixingKernel",
]

_sample_doc = """
        Samples a new route, using the previous route. You can call the instance to obtain the same effect.
    
        :param route: Previous route.
        :return: Modified route.
        """

_call_doc = """Alias for :py:method:`sample`"""


class Kernel(Protocol):
    """
    Protocol class used to generalise the behaviour of sampling new routes.
    It is used in the :py:class:`tsp.sa.SimulatedAnnealingTSP`.
    """
    rng: Generator
    """The generator of the instance."""

    def sample(self, route: Route) -> Route:
        ...

    def __call__(self, route: Route) -> Route:
        ...


class BaseKernel(metaclass=abc.ABCMeta):
    """
    Abstract class used to generalise the behaviour of sampling new routes.
    It is used in the :py:class:`tsp.sa.SimulatedAnnealingTSP`.
    """
    rng: Generator
    """The generator of the instance."""

    def __init__(self, seed: _GeneratorLike = None):
        self.rng: Generator = np.random.default_rng(seed)

    @abc.abstractmethod
    def sample(self, route: Route) -> Route:
        ...

    def __call__(self, route: Route) -> Route:
        return self.sample(route)


# Set the similar documentation
for _cls in [Kernel, BaseKernel]:
    _cls.sample.__doc__ = _sample_doc
    _cls.__call__.__doc__ = _call_doc


class SwapKernel(BaseKernel):
    r"""
    Kernel that performs a swap between two cities on a route.

    e.g.: For a route ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``, if ``i=3`` and ``j=7`` are selected as indices to swap,
    the new route would be ``[0, 1, 2, 7, 4, 5, 6, 3, 8, 9]``.
    """

    @staticmethod
    def _sample(route: Route, i: int, j: int) -> Route:
        """To have a deterministic sampling. Used for testing."""
        new_route_list = copy.copy(route.route)
        new_route_list[i], new_route_list[j] = new_route_list[j], new_route_list[i]
        return Route(new_route_list)

    def sample(self, route: Route) -> Route:
        # Use route length directly
        i, j = self.rng.choice(len(route)-1, size=2, replace=False) + 1
        return self._sample(route, i, j)


class ReversionKernel(BaseKernel):
    r"""
    Kernel that performs a reversion on a route.

    e.g.: For a route ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``, if ``i=3`` and ``j=7`` are selected as indices to reverse
    the route, the new route would be ``[0, 1, 2, 7, 6, 5, 4, 3, 8, 9]``.
    """

    @staticmethod
    def _sample(route: Route, i: int, j: int) -> Route:
        i, j = min(i, j), max(i, j)
        r = route.route
        if i == 0:
            new_r = r[j::-1] + r[j+1:]
        else:
            new_r = r[:i] + r[j:i-1:-1] + r[j+1:]
        return Route(new_r)

    def sample(self, route: Route) -> Route:
        i, j = self.rng.choice(len(route)-1, size=2, replace=False) + 1
        return self._sample(route, i, j)


class InsertionKernel(BaseKernel):
    r"""
    Kernel that performs an insertion on a route.

    e.g.: For a route ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``, if ``i=3`` and ``j=7`` are selected as indices to insert
    the city 3 into the place 7, the new route would be ``[0, 1, 2, 4, 5, 6, 7, 3, 8, 9]``.
    """

    @staticmethod
    def _sample(route: Route, i, j) -> Route:
        new_route_list = copy.copy(route.route)
        new_route_list.insert(j, new_route_list.pop(i))
        return Route(new_route_list)

    def sample(self, route: Route) -> Route:
        i, j = self.rng.choice(len(route)-1, size=2, replace=False) + 1
        return self._sample(route, i, j)


class RandomWalkKernel(BaseKernel):
    """
    Kernel that performs a random walk over possible path combinations. Returns a different path than the one
    initially given.
    """

    def sample(self, route: Route) -> Route:
        len_route = len(route)
        # We need to ensure we return a new object with different content
        # For small routes this loop might be slow but guarantees change
        current_route_list = route.route
        
        # Start with a copy
        new_route_list = list(current_route_list)
        
        while new_route_list == current_route_list:
             # Random permutation of 1..N
            perm = self.rng.choice(len_route-1, size=len_route-1, replace=False) + 1
            new_route_list = current_route_list[:1] + [current_route_list[i] for i in perm]
            
        return Route(new_route_list)


class MixingKernel(BaseKernel):
    r"""
    It is a mixing of the kernels. For example, if we have the kernels :math:`K_1`, :math:`K_2` and :math:`K_3`
    with the weights :math:`\{p_i\}_{i=1}^{3}`, then the resulting new kernel would be
    :math:`K = p_1 K_1 + p_2 K_2 + p_3 K_3`.
    """
    def __init__(
            self,
            kernels: Sequence[Kernel] | Sequence[tuple[float, Kernel]],
            probs: Optional[Sequence[float]] = None,
            seed: _GeneratorLike = None,
    ):
        super().__init__(seed)
        self._len_kernels = len(kernels)

        if probs is not None and len(probs) != len(kernels):
            raise ValueError("The lengths of 'kernels' and 'probs' must be equals.")

        # Array of probabilities
        if probs is None:
            # Check if kernels are tuples
            if kernels and isinstance(kernels[0], tuple):
                 # mypy help: assuming list of tuples
                 # type ignore for quick fix or proper casting
                 kp_list = t.cast(list[tuple[float, Kernel]], kernels)
                 probs = [p for p, k in kp_list]
                 kernels = [k for p, k in kp_list]
            else:
                 # Should probably error if probs is None and kernels are not tuples with probs?
                 # Assuming equal prob if not provided? Original code didn't handle this explicitly well if mixed types passed
                 # But let's assume if probs is None, it expects tuples.
                 pass

        # If still None (implied equal or failed partial logic above), raise or handle. 
        # Original code:
        # if probs is None:
        #    probs = [p for p, k in kernels]
        #    kernels = [k for p, k in kernels]
        
        # Let's clean up initialization logic
        final_kernels = []
        final_probs = []
        
        if probs is None:
             # Expect list of tuples
             for item in kernels:
                 if isinstance(item, tuple):
                     final_probs.append(item[0])
                     final_kernels.append(item[1])
                 else:
                     raise ValueError("If probs is None, kernels must be tuples of (prob, kernel)")
        else:
            final_kernels = list(kernels)
            final_probs = list(probs)

        probs_arr = np.asarray(final_probs) + _EPS * len(final_probs)
        probs_arr = probs_arr / np.sum(probs_arr)

        # Ordering the array of probabilities
        indices = []
        candidates = {i: probs_arr[i] for i in range(len(probs_arr))}
        while candidates:
            argmax_candidate = max(candidates, key=candidates.get)
            indices.append(argmax_candidate)
            del candidates[argmax_candidate]

        self.probs: np.ndarray = probs_arr.take(indices)
        self.kernels: list[Kernel] = [final_kernels[i] for i in indices]

        # Auxiliary variables
        self._cumsum_p: np.ndarray = np.concatenate([[0.], np.cumsum(self.probs)])

    def _choose_kernel(self, u: float) -> Kernel:
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
        return kernel # type: ignore

    def _sample(self, route: Route, u: float) -> Route:
        kernel = self._choose_kernel(u)
        return kernel.sample(route)

    def sample(self, route: Route) -> Route:
        u = self.rng.uniform()
        return self._sample(route, u)
