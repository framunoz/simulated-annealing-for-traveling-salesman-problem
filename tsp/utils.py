import functools

import numpy as np
from numpy.random import Generator

_Route = list[int]
_GeneratorLike = None | int | Generator

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
