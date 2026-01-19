using Test
using Random

const PROJECT_ROOT = joinpath(@__DIR__, "..", "..", "..")
const SRCJL_DIR = joinpath(PROJECT_ROOT, "srcjl")

if !isdefined(Main, :TspJulia)
    include(joinpath(SRCJL_DIR, "TspJulia.jl"))
end

using .TspJulia.Elements
using .TspJulia.Kernels

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
