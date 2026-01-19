using Test
using Random

# Adjusting path to point to `srcjl`
const PROJECT_ROOT = joinpath(@__DIR__, "..", "..", "..")
const SRCJL_DIR = joinpath(PROJECT_ROOT, "srcjl")

# Include the main module
if !isdefined(Main, :TspJulia)
    include(joinpath(SRCJL_DIR, "TspJulia.jl"))
end

using .TspJulia
using .TspJulia.Elements
using .TspJulia.Kernels

@testset "Kernels Tests" begin

    @testset "SwapKernel" begin
        # Seed 42 implies deterministic RNG
        kernel = SwapKernel(42)
        route = Route(collect(0:9))

        # In Python: _sample(route, 3, 7)
        # Julia indices: 3->4, 7->8 (1-based)
        swapped_route = _sample_swap(route, 4, 8)

        # Expected from Python: [0, 1, 2, 7, 4, 5, 6, 3, 8, 9]
        expected_route = Route([0, 1, 2, 7, 4, 5, 6, 3, 8, 9])

        @test swapped_route.route == expected_route.route

        # Verify original not modified
        @test route.route == collect(0:9)
    end

    @testset "ReversionKernel" begin
        kernel = ReversionKernel(42)
        route = Route(collect(0:9))

        # Test 1: i=3, j=7 (Python) -> i=4, j=8 (Julia)
        reversed_route1 = _sample_reversion(route, 4, 8)
        expected_route1 = Route([0, 1, 2, 7, 6, 5, 4, 3, 8, 9])
        @test reversed_route1.route == expected_route1.route

        # Test 2: i=7, j=3 (Python) -> i=8, j=4 (Julia)
        reversed_route2 = _sample_reversion(route, 8, 4)
        @test reversed_route2.route == expected_route1.route

        # Test 3: i=0, j=5 (Python) -> i=1, j=6 (Julia)
        reversed_route3 = _sample_reversion(route, 1, 6)
        expected_route3 = Route([5, 4, 3, 2, 1, 0, 6, 7, 8, 9])
        @test reversed_route3.route == expected_route3.route

        # Test 4: i=5, j=9 (Python) -> i=6, j=10 (Julia)
        reversed_route4 = _sample_reversion(route, 6, 10)
        expected_route4 = Route([0, 1, 2, 3, 4, 9, 8, 7, 6, 5])
        @test reversed_route4.route == expected_route4.route

        # Verify original not modified
        @test route.route == collect(0:9)
    end

    @testset "InsertionKernel" begin
        kernel = InsertionKernel(42)
        route = Route(collect(0:9))

        # Test 1: i=3, j=7 (Python) -> i=4, j=8 (Julia)
        inserted_route1 = _sample_insertion(route, 4, 8)
        expected_route1 = Route([0, 1, 2, 4, 5, 6, 7, 3, 8, 9])
        @test inserted_route1.route == expected_route1.route

        # Test 2: i=7, j=3 (Python) -> i=8, j=4 (Julia)
        inserted_route2 = _sample_insertion(route, 8, 4)
        expected_route2 = Route([0, 1, 2, 7, 3, 4, 5, 6, 8, 9])
        @test inserted_route2.route == expected_route2.route

        # Test 3: i=0, j=5 (Python) -> i=1, j=6 (Julia)
        inserted_route3 = _sample_insertion(route, 1, 6)
        expected_route3 = Route([1, 2, 3, 4, 5, 0, 6, 7, 8, 9])
        @test inserted_route3.route == expected_route3.route

        # Test 4: i=5, j=9 (Python) -> i=6, j=10 (Julia)
        inserted_route4 = _sample_insertion(route, 6, 10)
        expected_route4 = Route([0, 1, 2, 3, 4, 6, 7, 8, 9, 5])
        @test inserted_route4.route == expected_route4.route

        # Verify original not modified
        @test route.route == collect(0:9)
    end

    @testset "RandomWalkKernel" begin
        kernel = RandomWalkKernel(42)
        route = Route([0, 1, 2])

        for _ = 1:20
            new_route = sample(kernel, route)
            @test new_route.route != [0, 1, 2]
            @test new_route.route != [2, 0, 1]
            @test new_route.route != [1, 2, 0]
        end
    end

    @testset "MixingKernel" begin
        k_swap = SwapKernel(42)
        k_rev = ReversionKernel(42)
        k_ins = InsertionKernel(42)
        kernels = [k_swap, k_rev, k_ins]

        @testset "_choose_kernel" begin
            probs = [0.3, 0.2, 0.5]
            mixing = MixingKernel(kernels, probs)

            @test Kernels._choose_kernel(mixing, 0.0) == k_ins
            @test Kernels._choose_kernel(mixing, 0.4) == k_ins
            @test Kernels._choose_kernel(mixing, 0.5) == k_swap
            @test Kernels._choose_kernel(mixing, 0.7) == k_swap
            @test Kernels._choose_kernel(mixing, 0.8) == k_rev
            @test Kernels._choose_kernel(mixing, 0.99) == k_rev
            @test Kernels._choose_kernel(mixing, 1.0) == k_rev
        end

        @testset "Edge Cases" begin
            # Uniform probabilities
            mixing_uniform = MixingKernel(kernels, [1 / 3, 1 / 3, 1 / 3])
            @test length(unique(mixing_uniform.probs)) == 1

            # Single kernel with prob 1
            mixing_single = MixingKernel(kernels, [1.0, 0.0, 0.0])
            @test Kernels._choose_kernel(mixing_single, 0.5) == k_swap
            @test Kernels._choose_kernel(mixing_single, 0.0) == k_swap
            @test Kernels._choose_kernel(mixing_single, 0.99) == k_swap

            # Probabilities that sum to 1
            mixing_sum1 = MixingKernel(kernels, [1, 2, 7])
            @test sum(mixing_sum1.probs) ≈ 1.0
        end

        @testset "Integration Sample" begin
            mixing = MixingKernel(kernels, [1.0, 0.0, 0.0], 42)
            route = Route(collect(0:9))
            k_swap_ref = SwapKernel(42)
            @test sample(mixing, route) == sample(k_swap_ref, route)
        end
    end

end
