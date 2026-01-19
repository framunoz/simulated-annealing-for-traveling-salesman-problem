using Test

const PROJECT_ROOT = joinpath(@__DIR__, "..", "..", "..")
const SRCJL_DIR = joinpath(PROJECT_ROOT, "srcjl")

if !isdefined(Main, :TspJulia)
    include(joinpath(SRCJL_DIR, "TspJulia.jl"))
end

using .TspJulia.Elements

@testset "Route Tests" begin
    r_vec = [1, 2, 3, 4]
    r = Route(r_vec)

    @test r.route == Int16[1, 2, 3, 4]
    @test length(r) == 4
    @test r[1] == 1
    @test r[4] == 4
    
    # Test copy
    r_copy = copy(r)
    @test r_copy == r
    @test r_copy !== r 
    
    # Test mutation
    r[1] = 10
    @test r[1] == 10
    @test r.route[1] == 10
    
    # Test equality
    r2 = Route([10, 2, 3, 4])
    @test r == r2
    
    # Test range indexing
    @test r[2:3] == Int16[2, 3]
    
    # Test range setting
    r[2:3] = [20, 30]
    @test r.route == Int16[10, 20, 30, 4]
end
