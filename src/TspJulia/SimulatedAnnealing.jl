module SimulatedAnnealing

export AbstractSimulatedAnnealing, SimulatedAnnealingTSP, run_sa, SAStatsProxy

using Random
using ..Elements
using ..Kernels
using ..ScheduleFns

abstract type AbstractSimulatedAnnealing end

get_cities(sa::AbstractSimulatedAnnealing)::Cities =
    throw(NotImplementedError("get_cities not implemented for $(typeof(sa))"))
get_kernel(sa::AbstractSimulatedAnnealing)::AbstractKernel =
    throw(NotImplementedError("get_kernel not implemented for $(typeof(sa))"))
get_n_iter(sa::AbstractSimulatedAnnealing)::Int =
    throw(NotImplementedError("get_n_iter not implemented for $(typeof(sa))"))
get_cooling_schedule(sa::AbstractSimulatedAnnealing)::Function =
    throw(NotImplementedError("get_cooling_schedule not implemented for $(typeof(sa))"))
get_early_stop(sa::AbstractSimulatedAnnealing)::Bool =
    throw(NotImplementedError("get_early_stop not implemented for $(typeof(sa))"))
get_stop_after(sa::AbstractSimulatedAnnealing)::Int =
    throw(NotImplementedError("get_stop_after not implemented for $(typeof(sa))"))


initial_route(sa::AbstractSimulatedAnnealing)::Route = Route(1:length(get_cities(sa)))

sample_from_kernel(sa::AbstractSimulatedAnnealing, route::Route)::Route =
    sample(get_kernel(sa), route)

compute_value(sa::AbstractSimulatedAnnealing, route::Route)::Float64 =
    total_distance(get_cities(sa), route)

compute_log_acceptance_prob(
    sa::AbstractSimulatedAnnealing,
    route::Route,
    new_route::Route,
    k::Int,
)::Float64 =
    (-compute_value(sa, new_route) + compute_value(sa, route)) /
    (get_cooling_schedule(sa)(k) + eps(Float64))

select_route(
    sa::AbstractSimulatedAnnealing,
    route::Route,
    new_route::Route,
    log_accept_prob::Float64,
)::Route = log_accept_prob > 0 || rand() < exp(log_accept_prob) ? new_route : route

stop_criteria(sa::AbstractSimulatedAnnealing, k::Int)::Bool =
    get_early_stop(sa) && k > get_stop_after(sa)


function run_sa(sa::AbstractSimulatedAnnealing)::Route
    route = initial_route(sa)
    route_value = compute_value(sa, route)
    best_route = route
    best_value = route_value
    for k = 1:get_n_iter(sa)
        # route = step_sa(sa, route, k)
        new_route = sample_from_kernel(sa, route)
        log_accept_prob = compute_log_acceptance_prob(sa, route, new_route, k)
        route = select_route(sa, route, new_route, log_accept_prob)
        route_value = compute_value(sa, route)
        if route_value < best_value
            best_route = route
            best_value = route_value
        end
        if stop_criteria(sa, k)
            break
        end
    end
    return best_route
end


struct SimulatedAnnealingTSP{K<:AbstractKernel} <: AbstractSimulatedAnnealing
    cities::Cities
    kernel::K
    n_iter::Int
    cooling_schedule::Function
    early_stop::Bool
    stop_after::Int

    function SimulatedAnnealingTSP(
        cities::Cities,
        kernel::K,
        n_iter::Int = 1_000_000,
        cooling_schedule::Function = exponential_cooling_schedule(100, 0.99),
        early_stop::Bool = true,
        stop_after::Int = 10_000,
    ) where {K<:AbstractKernel}
        return new{K}(cities, kernel, n_iter, cooling_schedule, early_stop, stop_after)
    end
end

get_cities(sa::SimulatedAnnealingTSP)::Cities = sa.cities
get_kernel(sa::SimulatedAnnealingTSP)::AbstractKernel = sa.kernel
get_n_iter(sa::SimulatedAnnealingTSP)::Int = sa.n_iter
get_cooling_schedule(sa::SimulatedAnnealingTSP)::Function = sa.cooling_schedule
get_early_stop(sa::SimulatedAnnealingTSP)::Bool = sa.early_stop
get_stop_after(sa::SimulatedAnnealingTSP)::Int = sa.stop_after


mutable struct SAStatsProxy <: AbstractSimulatedAnnealing
    sa::SimulatedAnnealingTSP
    log_acceptance_probs::Vector{Float64}
    n_accept::Int64
    acceptance_ratio::Vector{Float64}
    values::Vector{Float16}

    SAStatsProxy(sa::SimulatedAnnealingTSP) = new(sa, [], 0, [], [])
end

get_cities(sa::SAStatsProxy)::Cities = sa.sa.cities
get_kernel(sa::SAStatsProxy)::AbstractKernel = sa.sa.kernel
get_n_iter(sa::SAStatsProxy)::Int = sa.sa.n_iter
get_cooling_schedule(sa::SAStatsProxy)::Function = sa.sa.cooling_schedule
get_early_stop(sa::SAStatsProxy)::Bool = sa.sa.early_stop
get_stop_after(sa::SAStatsProxy)::Int = sa.sa.stop_after

function Base.show(io::IO, sa::SAStatsProxy)
    repr =
        "SAStatsProxy(" *
        string(sa.sa) *
        ", log_acceptance_probs=$(length(sa.log_acceptance_probs)), n_accept=$(sa.n_accept), acceptance_ratio=$(length(sa.acceptance_ratio)), values=$(length(sa.values))"
    print(io, repr)
end


function sample_from_kernel(sa::SAStatsProxy, route::Route)
    new_route = sample_from_kernel(sa.sa, route)
    push!(sa.values, compute_value(sa, new_route))
    return new_route
end

function compute_log_acceptance_prob(
    sa::SAStatsProxy,
    route::Route,
    new_route::Route,
    k::Int,
)
    log_accept_prob = compute_log_acceptance_prob(sa.sa, route, new_route, k)
    push!(sa.log_acceptance_probs, log_accept_prob)
    return log_accept_prob
end

function select_route(
    sa::SAStatsProxy,
    route::Route,
    new_route::Route,
    log_accept_prob::Float64,
)
    if log_accept_prob > 0 || rand() < exp(log_accept_prob)
        sa.n_accept += 1
        return new_route
    else
        return route
    end
end


end  # module SimulatedAnnealing
