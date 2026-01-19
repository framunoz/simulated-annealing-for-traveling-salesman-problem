using Test
using Random

const PROJECT_ROOT = joinpath(@__DIR__, "..", "..", "..")
const SRCJL_DIR = joinpath(PROJECT_ROOT, "srcjl")

if !isdefined(Main, :TspJulia)
    include(joinpath(SRCJL_DIR, "TspJulia.jl"))
end

using .TspJulia.Elements
using .TspJulia.Kernels

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
