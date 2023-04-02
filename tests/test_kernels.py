import copy
from unittest import TestCase

from tsp.kernels import *


class TestSwapKernelTSP(TestCase):
    def setUp(self) -> None:
        self.kernel = SwapKernelTSP(42)

    def test_sample(self):
        route = [i for i in range(10)]

        swapped_route = self.kernel._sample(route, 3, 7)
        expected_route = [0, 1, 2, 7, 4, 5, 6, 3, 8, 9]

        self.assertEqual(
            expected_route, swapped_route,
            "The '_sample' method is not working. It does not swapping as expected."
        )
        self.assertEqual(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], route,
            "The original route has been modified by the '_sample' method."
        )


class TestReversionKernelTSP(TestCase):
    def setUp(self) -> None:
        self.kernel = ReversionKernelTSP(42)

    def test_sample(self):
        route = list(range(10))

        reversed_route1 = self.kernel._sample(route, 3, 7)
        expected_route1 = [0, 1, 2, 7, 6, 5, 4, 3, 8, 9]
        self.assertEqual(
            expected_route1, reversed_route1,
            "The '_sample' method is not working. It does not reverse as expected."
        )

        reversed_route2 = self.kernel._sample(route, 7, 3)
        expected_route2 = [0, 1, 2, 7, 6, 5, 4, 3, 8, 9]
        self.assertEqual(
            expected_route2, reversed_route2,
            "The '_sample' method is not working. It is not index-agnostic."
        )

        reversed_route3 = self.kernel._sample(route, 0, 5)
        expected_route3 = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9]
        self.assertEqual(
            expected_route3, reversed_route3,
            "The '_sample' method is not working index i=0."
        )

        reversed_route4 = self.kernel._sample(route, 5, 9)
        expected_route4 = [0, 1, 2, 3, 4, 9, 8, 7, 6, 5]
        self.assertEqual(
            expected_route4, reversed_route4,
            "The '_sample' method is not working index j=last."
        )

        self.assertEqual(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], route,
            "The original route has been modified by the '_sample' method."
        )


class TestInsertionKernelTSP(TestCase):
    def setUp(self) -> None:
        self.kernel = InsertionKernelTSP(42)

    def test_sample(self):
        route = list(range(10))

        inserted_route1 = self.kernel._sample(route, 3, 7)
        expected_route1 = [0, 1, 2, 4, 5, 6, 3, 7, 8, 9]
        self.assertEqual(
            expected_route1, inserted_route1,
            "The '_sample' method is not working. Test 1."
        )

        inserted_route2 = self.kernel._sample(route, 7, 3)
        expected_route2 = [0, 1, 2, 7, 3, 4, 5, 6, 8, 9]
        self.assertEqual(
            expected_route2, inserted_route2,
            "The '_sample' method is not working. Test 2."
        )

        inserted_route3 = self.kernel._sample(route, 0, 5)
        expected_route3 = [1, 2, 3, 4, 0, 5, 6, 7, 8, 9]
        self.assertEqual(
            expected_route3, inserted_route3,
            "The '_sample' method is not working index i=0."
        )

        inserted_route4 = self.kernel._sample(route, 5, 9)
        expected_route4 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 5]
        self.assertEqual(
            expected_route4, inserted_route4,
            "The '_sample' method is not working index j=last."
        )

        self.assertEqual(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], route,
            "The original route has been modified by the '_sample' method."
        )
