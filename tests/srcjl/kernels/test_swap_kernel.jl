using Test
using Random

const PROJECT_ROOT = joinpath(@__DIR__, "..", "..", "..")
const SRCJL_DIR = joinpath(PROJECT_ROOT, "srcjl")

if !isdefined(Main, :TspJulia)
    include(joinpath(SRCJL_DIR, "TspJulia.jl"))
end

using .TspJulia.Elements
using .TspJulia.Kernels

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
