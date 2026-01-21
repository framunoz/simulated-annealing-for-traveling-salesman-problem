
import functools
from unittest import TestCase

import numpy as np

from tsp.py.elements import Route
from tsp.py.kernels import *


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



class TestSwapKernel(TestCase):
    def setUp(self) -> None:
        self.kernel = SwapKernel(42)

    def test_sample(self):
        route = Route([i for i in range(10)])

        swapped_route = self.kernel._sample(route, 3, 7)
        expected_route_list = [0, 1, 2, 7, 4, 5, 6, 3, 8, 9]

        self.assertEqual(
            expected_route_list, swapped_route.route,
            "The '_sample' method is not working. It does not swapping as expected."
        )
        self.assertEqual(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], route.route,
            "The original route has been modified by the '_sample' method."
        )


class TestReversionKernel(TestCase):
    def setUp(self) -> None:
        self.kernel = ReversionKernel(42)

    def test_sample(self):
        route = Route(list(range(10)))

        reversed_route1 = self.kernel._sample(route, 3, 7)
        expected_route1 = [0, 1, 2, 7, 6, 5, 4, 3, 8, 9]
        self.assertEqual(
            expected_route1, reversed_route1.route,
            "The '_sample' method is not working. It does not reverse as expected."
        )

        reversed_route2 = self.kernel._sample(route, 7, 3)
        expected_route2 = [0, 1, 2, 7, 6, 5, 4, 3, 8, 9]
        self.assertEqual(
            expected_route2, reversed_route2.route,
            "The '_sample' method is not working. It is not index-agnostic."
        )

        reversed_route3 = self.kernel._sample(route, 0, 5)
        expected_route3 = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9]
        self.assertEqual(
            expected_route3, reversed_route3.route,
            "The '_sample' method is not working index i=0."
        )

        reversed_route4 = self.kernel._sample(route, 5, 9)
        expected_route4 = [0, 1, 2, 3, 4, 9, 8, 7, 6, 5]
        self.assertEqual(
            expected_route4, reversed_route4.route,
            "The '_sample' method is not working index j=last."
        )

        self.assertEqual(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], route.route,
            "The original route has been modified by the '_sample' method."
        )


class TestInsertionKernel(TestCase):
    def setUp(self) -> None:
        self.kernel = InsertionKernel(42)

    def test_sample(self):
        route = Route(list(range(10)))

        inserted_route1 = self.kernel._sample(route, 3, 7)
        expected_route1 = [0, 1, 2, 4, 5, 6, 7, 3, 8, 9]
        self.assertEqual(
            expected_route1, inserted_route1.route,
            "The '_sample' method is not working. Test 1."
        )

        inserted_route2 = self.kernel._sample(route, 7, 3)
        expected_route2 = [0, 1, 2, 7, 3, 4, 5, 6, 8, 9]
        self.assertEqual(
            expected_route2, inserted_route2.route,
            "The '_sample' method is not working. Test 2."
        )

        inserted_route3 = self.kernel._sample(route, 0, 5)
        expected_route3 = [1, 2, 3, 4, 5, 0, 6, 7, 8, 9]
        self.assertEqual(
            expected_route3, inserted_route3.route,
            "The '_sample' method is not working index i=0."
        )

        inserted_route4 = self.kernel._sample(route, 5, 9)
        expected_route4 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 5]
        self.assertEqual(
            expected_route4, inserted_route4.route,
            "The '_sample' method is not working index j=last."
        )

        self.assertEqual(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], route.route,
            "The original route has been modified by the '_sample' method."
        )


class TestRandomWalkKernel(TestCase):
    def setUp(self) -> None:
        self.kernel = RandomWalkKernel(42)

    @repeat(20)
    def test_sample(self):
        route = Route([0, 1, 2])
        # The possible permutations of this route are:
        # [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
        # Removing circular permutations: [0, 1, 2] -> [2, 0, 1] -> [1, 2, 0]
        # And the possible options are: [0, 2, 1]; [1, 0, 2]; [2, 1, 0]
        new_route = self.kernel(route)
        self.assertNotEqual(
            [0, 1, 2], new_route.route,
            "The 'new_route' must be different of the original route."
        )
        self.assertNotEqual(
            [2, 0, 1], new_route.route,
            "The 'new_route' must be different of the original route permutation 1."
        )
        self.assertNotEqual(
            [1, 2, 0], new_route.route,
            "The 'new_route' must be different of the original route permutation 2."
        )


class TestMixingKernel(TestCase):
    def setUp(self) -> None:
        self.kernel = MixingKernel(
            [SwapKernel(42), ReversionKernel(42), InsertionKernel(42)],
            [0.3, 0.2, 0.5], seed=42
        )
        self.kernel2 = MixingKernel(
            [(0.3, SwapKernel(42)),
             (0.2, ReversionKernel(42)),
             (0.5, InsertionKernel(42))],
            seed=42,
        )

    @repeat(20)
    def test_sample(self):
        # Testing the different ways to construct a kernel
        kernel = self.kernel if np.random.uniform() <= 0.5 else self.kernel2

        # Note: _choose_kernel is internal, depends on impl. 
        # But we can test it if we know the RNG state or property
        
        # Testing _choose_kernel directly with fixed values
        # Intervals of probabilities for u ~ Uniform[0, 1]
        # probs order is sorted by value? No, my implementation sorts them?
        # Original implementation sorted them. My implementation sorts them.
        # Original:
        # candidates = {i: probs[i] ...}
        # argmax_candidate = max(candidates, key=candidates.get)
        # So it sorts indices by probability descending.
        # probs: [0.3, 0.2, 0.5]
        # 0: 0.3, 1: 0.2, 2: 0.5
        # sorted: index 2 (0.5), index 0 (0.3), index 1 (0.2)
        # cumsum: 0, 0.5, 0.5+0.3=0.8, 0.8+0.2=1.0
        # Intervals: [0, 0.5) -> index 2 (Insertion)
        #            [0.5, 0.8) -> index 0 (Swap)
        #            [0.8, 1.0) -> index 1 (Reversion)
        
        # u = 0 -> InsertionKernel
        kernel1 = kernel._choose_kernel(0)
        expected_class = InsertionKernel
        self.assertIsInstance(
            kernel1, expected_class,
            "The method '_choose_kernel' does not work with value 0."
        )
        #  u in (0.0, 0.5) -> InsertionKernel
        kernel2 = kernel._choose_kernel(0.25)
        expected_class = InsertionKernel
        self.assertIsInstance(
            kernel2, expected_class,
            "The method '_choose_kernel' does not work for 0.25"
        )
        #  u in (0.5, 0.8) -> SwapKernel
        kernel3 = kernel._choose_kernel(0.65)
        expected_class = SwapKernel
        self.assertIsInstance(
            kernel3, expected_class,
            "The method '_choose_kernel' does not work for 0.65"
        )
        #  u in (0.8, 1.0) -> ReversionKernel
        kernel4 = kernel._choose_kernel(0.9)
        expected_class = ReversionKernel
        self.assertIsInstance(
            kernel4, expected_class,
            "The method '_choose_kernel' does not work for 0.9"
        )
