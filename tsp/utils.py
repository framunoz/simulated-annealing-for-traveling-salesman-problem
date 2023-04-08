import functools
from typing import Iterable

import numpy as np
from numpy.random import Generator

_Route = list[int]
_GeneratorLike = None | int | Generator
_ArrayLike = np.ndarray | Iterable

_EPS = np.finfo(np.float64).eps


def repeat(times: int):
    """
    Decorator for repeated tests.

    :param times: The number of times to repeat the test.
    :return: the function wrapped.
    """
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            value = None
            for _ in range(times):
                value = func(*args, **kwargs)
            return value

        return wrapper

    return decorator_repeat


# noinspection PyPep8Naming
def logarithmic_cooling_schedule(k, T_0=25, k_0=1) -> float:
    r"""
    The logarithmic cooling schedule is

    .. math::
        T_k = T_0 \cdot \ln(k_0 + 1) / \ln(k + 1 + k_0)

    :param int k: The number of the iteration at the temperature :math:`T_k`.
    :param float T_0: Initial temperature. Default is 25
    :param float k_0: Iteration phase shift. Default is 1
    :return: The temperature at iteration :math:`k`.
    """
    return (T_0 * np.log(k_0)) / (np.log(k + k_0))


# noinspection PyPep8Naming
def exponential_cooling_schedule(k, T_0=100, rho=0.99) -> float:
    r"""
    The exponential cooling schedule is

    .. math::
        T_k = \rho^k \cdot T_0

    :param int k: The number of the iteration at the temperature :math:`T_k`.
    :param float T_0: Initial temperature. Default is 100.
    :param rho: The decay parameter. The default is 0.99. It is recommended to increase this amount to a value of 0.999 or 0.9999...
    :return: The temperature at iteration :math:`k`.
    """
    return rho ** k * T_0
