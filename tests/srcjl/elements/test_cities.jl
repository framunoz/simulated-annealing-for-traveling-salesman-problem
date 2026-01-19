using Test

const PROJECT_ROOT = joinpath(@__DIR__, "..", "..", "..")
const SRCJL_DIR = joinpath(PROJECT_ROOT, "srcjl")

if !isdefined(Main, :TspJulia)
    include(joinpath(SRCJL_DIR, "TspJulia.jl"))
end

using .TspJulia.Elements

@testset "Cities Tests" begin
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    p3 = Point(6, 8)
    
    c = Cities([p1, p2, p3])
    
    @test length(c) == 3
    @test c[1] == p1
    @test c[3] == p3
    
    # Test compute_distance_matrix
    dm = compute_distance_matrix(c)
    @test size(dm) == (3, 3)
    @test dm[1, 1] == 0.0
    @test dm[1, 2] == 5.0 # 3-4-5 triangle
    @test dm[2, 3] == 5.0
    @test dm[1, 3] == 10.0
    
    # Test total_distance
    r = Route([1, 2, 3])
    # 1->2 (5.0) + 2->3 (5.0) + 3->1 (10.0) = 20.0
    td = total_distance(c, r)
    @test td ≈ 20.0
    
    # Test subsetting with route
    c_sub = c[Route([1, 3])]
    @test length(c_sub) == 2
    @test c_sub[1] == p1
    @test c_sub[2] == p3
end
