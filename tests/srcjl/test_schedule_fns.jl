using Test

const PROJECT_ROOT = joinpath(@__DIR__, "..", "..")
const SRCJL_DIR = joinpath(PROJECT_ROOT, "srcjl")

if !isdefined(Main, :TspJulia)
    include(joinpath(SRCJL_DIR, "TspJulia.jl"))
end

using .TspJulia.ScheduleFns

@testset "Schedule Functions Tests" begin
    
    @testset "exponential_cooling_schedule" begin
        # T(k) = T0 * rho^k
        T0 = 100.0
        rho = 0.9
        schedule = exponential_cooling_schedule(T₀=T0, ρ=rho)
        
        @test schedule(0) ≈ T0
        @test schedule(1) ≈ T0 * rho
        @test schedule(2) ≈ T0 * rho^2
    end
    
    @testset "log_cooling_schedule" begin
        # T(k) = T0 * log(k0) / log(k + k0)
        T0 = 100.0
        k0 = 10.0
        schedule = log_cooling_schedule(T₀=T0, k₀=k0)
        
        @test schedule(0) ≈ T0
        # for k=k0, T(k0) = T0 * log(k0) / log(2*k0) = T0 * log(k0) / (log(2) + log(k0))
        expected_k0 = T0 * log(k0) / log(k0 + k0)
        @test schedule(k0) ≈ expected_k0
    end
    
    @testset "linear_increase_cooling_schedule" begin
        # T(k) = T0 + (T1 - T0) * k / n_iter
        T0 = 10.0
        T1 = 110.0
        n_iter = 100
        schedule = linear_increase_cooling_schedule(T₀=T0, T₁=T1, n_iter=n_iter)
        
        @test schedule(0) ≈ T0
        @test schedule(n_iter) ≈ T1
        @test schedule(50) ≈ 60.0 # Midpoint
    end
    
    @testset "linear_decrease_cooling_schedule" begin
        # T(k) = T0 - (T0 - T1) * k / n_iter
        T0 = 100.0
        T1 = 0.0
        n_iter = 100
        schedule = linear_decrease_cooling_schedule(T₀=T0, T₁=T1, n_iter=n_iter)
        
        @test schedule(0) ≈ T0
        @test schedule(n_iter) ≈ T1
        @test schedule(50) ≈ 50.0
    end

end
