
import itertools
from typing import Iterable, Literal, Optional, Protocol

import numpy as np
import scipy.spatial
from numpy.random import Generator

__all__ = [
    "cooling_schedule_function",
    "logarithmic_cooling_schedule",
    "exponential_cooling_schedule",
    "sample_coordinates",
    "distance_matrix",
    "_GeneratorLike",
    "_ArrayLike",
    "_LiteralStyles",
    "_EPS"
]

_GeneratorLike = None | int | Generator
_ArrayLike = np.ndarray | Iterable
_LiteralStyles = Literal["darkgrid", "whitegrid", "dark", "white", "ticks"]

_EPS = np.finfo(np.float64).eps

_call_doc = """
        Call method.
        
        :param int k: The number of the iteration at the temperature :math:`T_k`. 
        :return: The temperature at iteration :math:`k`.
        """


class cooling_schedule_function(Protocol):
    """
    Protocol for cooling schedule functions.
    """
    def __call__(self, k: int) -> float:
        ...


class logarithmic_cooling_schedule:
    r"""
    The logarithmic cooling schedule is

    .. math::
        T_k = T_0 \cdot \ln(k_0 + 1) / \ln(k + 1 + k_0)

    :param T_0: Initial temperature. Default is 25.
    :param k_0: Iteration phase shift. Default is 1
    """

    def __init__(self, T_0=25, k_0=1):
        self.T_0 = T_0
        self.k_0 = k_0

    def __call__(self, k: int) -> float:
        return (self.T_0 * np.log(self.k_0)) / (np.log(k + self.k_0))


class exponential_cooling_schedule:
    r"""
    The exponential cooling schedule is

    .. math::
        T_k = \rho^k \cdot T_0

    :param float T_0: Initial temperature. Default is 100.
    :param rho: The decay parameter. The default is 0.99. It is recommended to increase this amount to a value of 0.999
        or 0.9999...
    """
    def __init__(self, T_0=100, rho=0.99):
        self.T_0 = T_0
        self.rho = rho

    def __call__(self, k: int) -> float:
        return self.rho ** k * self.T_0


# Assign the similar documentation
for _cls in [cooling_schedule_function, logarithmic_cooling_schedule, exponential_cooling_schedule]:
    _cls.__call__.__doc__ = _call_doc


def sample_coordinates(
        n_samples: int,
        lb: Optional[int] = None,
        ub: Optional[int] = None,
        seed: _GeneratorLike = None
) -> np.ndarray:
    r"""
    Returns a sample of points along a grid of :math:`[lb, ub] \times [lb, up]`.

    :param n_samples: Number of samples.
    :param lb: The lower bound. Default is 0.
    :param ub: The upper bound. Default is n_samples.
    :param seed: The seed or generator.
    :return: The coordinates.
    """
    lb = lb if lb is not None else 0
    ub = ub if ub is not None else n_samples

    rng = np.random.default_rng(seed)

    return rng.choice(list(itertools.product(range(lb, ub + 1), repeat=2)), size=n_samples)


def distance_matrix(coords: _ArrayLike) -> np.ndarray:
    """
    Calculate the distance matrix from the coordinates.

    :param coords: Coordinate array of shape ``(n_coord, n_dim)``.
    :return: Distance matrix of shape ``(n_coord, n_coord)``
    """

    coords = np.asarray(coords)

    return scipy.spatial.distance_matrix(coords, coords)
