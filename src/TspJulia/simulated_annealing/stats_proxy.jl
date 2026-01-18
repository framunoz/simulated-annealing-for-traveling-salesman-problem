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
get_cooling_schedule(sa::SAStatsProxy)::Function = sa.sa.cooling_schedule
get_early_stop(sa::SAStatsProxy)::Bool = sa.sa.early_stop
get_stop_after(sa::SAStatsProxy)::Int = sa.sa.stop_after

function Base.show(io::IO, sa::SAStatsProxy)
    repr = "SAStatsProxy(log_acceptance_probs=$(length(sa.log_acceptance_probs)), n_accept=$(sa.n_accept), acceptance_ratio=$(length(sa.acceptance_ratio)), values=$(length(sa.values)))"
    return print(io, repr)
end

function sample_from_kernel(sa::SAStatsProxy, route::Elements.Route)::Elements.Route
    new_route = sample_from_kernel(sa.sa, route)
    push!(sa.values, compute_value(sa, new_route))
    return new_route
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

function select_route(
    sa::SAStatsProxy,
    route::Elements.Route,
    new_route::Elements.Route,
    log_accept_prob::Float64,
)::Elements.Route
    if log_accept_prob > 0 || rand() < exp(log_accept_prob)
        sa.n_accept += 1
        return new_route
    else
        return route
    end
end
