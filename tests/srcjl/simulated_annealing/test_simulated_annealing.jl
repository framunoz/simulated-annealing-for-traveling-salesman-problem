using Test
using Random

const PROJECT_ROOT = joinpath(@__DIR__, "..", "..", "..")
const SRCJL_DIR = joinpath(PROJECT_ROOT, "srcjl")

if !isdefined(Main, :TspJulia)
    include(joinpath(SRCJL_DIR, "TspJulia.jl"))
end

using .TspJulia.Elements
using .TspJulia.Kernels
using .TspJulia.SimulatedAnnealing
using .TspJulia.ScheduleFns
import .TspJulia.SimulatedAnnealing: get_n_iter, get_early_stop

@testset "Simulated Annealing Tests" begin
    
    # Setup simple problem
    cities = Cities([
        Point(0, 0),
        Point(1, 0),
        Point(1, 1),
        Point(0, 1)
    ])
    # Optimal route length is 4.0 (square perimeter)
    
    kernel = SwapKernel(42)
    
    @testset "Initialization" begin
        sa = SimulatedAnnealingTSP(
            cities, 
            kernel, 
            n_iter=100, 
            early_stop=false
        )
        
        @test get_n_iter(sa) == 100
        @test !get_early_stop(sa)
    end
    
    @testset "Execution" begin
        # Create an SA instance
        sa = SimulatedAnnealingTSP(
            cities, 
            kernel, 
            n_iter=1000, 
            early_stop=false,
            temperature = exponential_cooling_schedule(T₀=10.0, ρ=0.95)
        )
        
        # Run
        best_route, best_val = run_sa(sa)
        
        @test best_route isa Route
        @test best_val isa Float64
        @test length(best_route) == 4
        
        # Check if result is reasonable (at least not worse than worst possible)
        # Worst is diagonals: sqrt(2)*2 + 2 approx 4.828
        # Best is 4.0
        @test best_val >= 4.0
    end
    
    @testset "Stats Proxy" begin
        sa = SimulatedAnnealingTSP(
            cities, 
            kernel, 
            n_iter=100, 
            early_stop=false
        )
        proxy = SAStatsProxy(sa)
        
        best_route, best_val = run_sa(proxy)
        
        @test length(proxy.values) == 101
        @test length(proxy.log_acceptance_probs) == 100
        @test length(proxy.acceptance_ratio) == 100
        @test proxy.n_accept >= 0
    end
end
