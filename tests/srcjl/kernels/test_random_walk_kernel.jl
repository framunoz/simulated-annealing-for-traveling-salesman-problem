using Test
using Random

const PROJECT_ROOT = joinpath(@__DIR__, "..", "..", "..")
const SRCJL_DIR = joinpath(PROJECT_ROOT, "srcjl")

if !isdefined(Main, :TspJulia)
    include(joinpath(SRCJL_DIR, "TspJulia.jl"))
end

using .TspJulia.Elements
using .TspJulia.Kernels

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
