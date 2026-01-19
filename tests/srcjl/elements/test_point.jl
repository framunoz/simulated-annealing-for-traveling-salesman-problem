using Test

const PROJECT_ROOT = joinpath(@__DIR__, "..", "..", "..")
const SRCJL_DIR = joinpath(PROJECT_ROOT, "srcjl")

if !isdefined(Main, :TspJulia)
    include(joinpath(SRCJL_DIR, "TspJulia.jl"))
end

using .TspJulia.Elements

@testset "Point Tests" begin
    p1 = Point(1.0, 2.0)
    p2 = Point(3, 4) # Integer conversion

    @test p1.x == 1.0f0
    @test p1.y == 2.0f0
    @test p2.x == 3.0f0
    @test p2.y == 4.0f0

    @test string(p1) == "P(1.0, 2.0)"
    
    # Test distance
    d = distance(p1, p2)
    # sqrt((1-3)^2 + (2-4)^2) = sqrt(4 + 4) = sqrt(8) approx 2.8284
    @test d ≈ sqrt(8) atol=1e-4
end
