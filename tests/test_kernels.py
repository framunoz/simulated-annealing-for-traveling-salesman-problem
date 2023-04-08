from unittest import TestCase

from tsp.kernels import *
from tsp.utils import repeat


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


class TestRandomWalkKernelTSP(TestCase):
    def setUp(self) -> None:
        self.kernel = RandomWalkKernelTSP(42)

    @repeat(20)
    def test_sample(self):
        route = [0, 1, 2]
        # The possible permutations of this route are:
        # [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
        # Removing circular permutations: [0, 1, 2] -> [2, 0, 1] -> [1, 2, 0]
        # And the possible options are: [0, 2, 1]; [1, 0, 2]; [2, 1, 0]
        new_route = self.kernel(route)
        self.assertNotEqual(
            [0, 1, 2], new_route,
            "The 'new_route' must be different of the original route."
        )
        self.assertNotEqual(
            [2, 0, 1], new_route,
            "The 'new_route' must be different of the original route permutation 1."
        )
        self.assertNotEqual(
            [1, 2, 0], new_route,
            "The 'new_route' must be different of the original route permutation 2."
        )


class TestMixingKernelTSP(TestCase):
    def setUp(self) -> None:
        self.kernel = MixingKernelTSP(
            [SwapKernelTSP(42), ReversionKernelTSP(42), InsertionKernelTSP(42)],
            [0.3, 0.2, 0.5], seed=42
        )
        self.kernel2 = MixingKernelTSP(
            [(0.3, SwapKernelTSP(42)),
             (0.2, ReversionKernelTSP(42)),
             (0.5, InsertionKernelTSP(42))],
            seed=42,
        )

    @repeat(20)
    def test_sample(self):
        # Testing the different ways to construct a kernel
        kernel = self.kernel if np.random.uniform() <= 0.5 else self.kernel2

        # Intervals of probabilities for u ~ Uniform[0, 1]
        # u = 0 -> InsertionKernelTSP
        kernel1 = kernel._choose_kernel(0)
        expected_class = InsertionKernelTSP
        self.assertIsInstance(
            kernel1, expected_class,
            "The method '_choose_kernel' does not work with value 0. test1"
        )
        #  u in (0.0, 0.5) -> InsertionKernelTSP
        u = np.random.uniform(0.0, 0.5)
        kernel2 = kernel._choose_kernel(u)
        expected_class = InsertionKernelTSP
        self.assertIsInstance(
            kernel2, expected_class,
            "The method '_choose_kernel' does not work. test2"
        )
        #  u in (0.5, 0.8) -> SwapKernelTSP
        u = np.random.uniform(0.5, 0.8)
        kernel3 = kernel._choose_kernel(u)
        expected_class = SwapKernelTSP
        self.assertIsInstance(
            kernel3, expected_class,
            "The method '_choose_kernel' does not work. test3"
        )
        #  u in (0.8, 1.0) -> ReversionKernelTSP
        u = np.random.uniform(0.8, 1.0)
        kernel4 = kernel._choose_kernel(u)
        expected_class = ReversionKernelTSP
        self.assertIsInstance(
            kernel4, expected_class,
            "The method '_choose_kernel' does not work. test4"
        )
        #  u = 1 -> ReversionKernelTSP
        kernel5 = kernel._choose_kernel(1)
        expected_class = ReversionKernelTSP
        self.assertIsInstance(
            kernel5, expected_class,
            "The method '_choose_kernel' does not work. test5"
        )
