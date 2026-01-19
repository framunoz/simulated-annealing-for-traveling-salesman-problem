"""
    SAStatsProxy(sa)

A proxy wrapper for [`SimulatedAnnealingTSP`](@ref) that collects statistics during execution.

It tracks:

  - `log_acceptance_probs`: Logarithm of acceptance probabilities at each step.
  - `n_accept`: Total number of accepted moves.
  - `values`: Objective function values (route distances) at each step.

# Example

```jldoctest
julia> using TspJulia.Elements, TspJulia.Kernels, TspJulia.SimulatedAnnealing

julia> cities = Cities([Point(0, 0), Point(1, 1), Point(0, 1)]);

julia> sa = SimulatedAnnealingTSP(cities, SwapKernel(42));

julia> stats = SAStatsProxy(sa);

julia> isa(stats, AbstractSimulatedAnnealing)
true
```
"""
mutable struct SAStatsProxy <: AbstractSimulatedAnnealing
    sa::SimulatedAnnealingTSP
    log_acceptance_probs::Vector{Float64}
    n_accept::Int64
    acceptance_ratio::Vector{Float64}
    values::Vector{Float16}

    SAStatsProxy(sa::SimulatedAnnealingTSP) = new(sa, [], 0, [], [])
end

get_cities(sa::SAStatsProxy)::Elements.Cities = sa.sa.cities
get_kernel(sa::SAStatsProxy)::Kernels.AbstractKernel = sa.sa.kernel
get_n_iter(sa::SAStatsProxy)::Int = sa.sa.n_iter
get_temperature(sa::SAStatsProxy)::Function = sa.sa.temperature
get_early_stop(sa::SAStatsProxy)::Bool = sa.sa.early_stop
get_stop_after(sa::SAStatsProxy)::Int = sa.sa.stop_after

function Base.show(io::IO, sa::SAStatsProxy)
    if length(sa.values) == 0
        print(io, "SAStatsProxy()")
        return nothing
    end
    repr = "SAStatsProxy("
    repr *= "log_acceptance_probs=$(length(sa.log_acceptance_probs)), "
    repr *= "n_accept=$(sa.n_accept), "
    repr *= "acceptance_ratio=$(round(100 * sa.acceptance_ratio[end], digits=2))%, "
    repr *= "values=$(round(sa.values[end], digits=4))"
    repr *= ")"
    return print(io, repr)
end

callback_after_acceptance(sa::SAStatsProxy, k::Int) = sa.n_accept += 1

callback_after_iteration(sa::SAStatsProxy, k::Int) =
    push!(sa.acceptance_ratio, sa.n_accept / k)

function initial_route(sa::SAStatsProxy)::Tuple{Elements.Route, Float64}
    route, route_value = initial_route(sa.sa)
    push!(sa.values, route_value)
    return route, route_value
end

function sample_from_kernel(
    sa::SAStatsProxy,
    route::Elements.Route,
)::Tuple{Elements.Route, Float64}
    new_route, new_route_value = sample_from_kernel(sa.sa, route)
    push!(sa.values, new_route_value)
    return new_route, new_route_value
end

function compute_log_acceptance_prob(
    sa::SAStatsProxy,
    route::Elements.Route,
    new_route::Elements.Route,
    k::Int,
)::Float64
    log_accept_prob = compute_log_acceptance_prob(sa.sa, route, new_route, k)
    push!(sa.log_acceptance_probs, log_accept_prob)
    return log_accept_prob
end
