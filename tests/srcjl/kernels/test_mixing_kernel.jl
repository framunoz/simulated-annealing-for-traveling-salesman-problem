using Test
using Random

const PROJECT_ROOT = joinpath(@__DIR__, "..", "..", "..")
const SRCJL_DIR = joinpath(PROJECT_ROOT, "srcjl")

if !isdefined(Main, :TspJulia)
    include(joinpath(SRCJL_DIR, "TspJulia.jl"))
end

using .TspJulia.Elements
using .TspJulia.Kernels

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
