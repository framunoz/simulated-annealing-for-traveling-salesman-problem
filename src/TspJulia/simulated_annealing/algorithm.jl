initial_route(sa::AbstractSimulatedAnnealing)::Elements.Route =
    Elements.Route(1:length(get_cities(sa)))

sample_from_kernel(
    sa::AbstractSimulatedAnnealing,
    route::Elements.Route,
)::Elements.Route = sample(get_kernel(sa), route)

compute_value(sa::AbstractSimulatedAnnealing, route::Elements.Route)::Float64 =
    total_distance(get_cities(sa), route)

compute_log_acceptance_prob(
    sa::AbstractSimulatedAnnealing,
    route::Elements.Route,
    new_route::Elements.Route,
    k::Int,
)::Float64 =
    (-compute_value(sa, new_route) + compute_value(sa, route)) /
    (get_cooling_schedule(sa)(k) + eps(Float64))

select_route(
    sa::AbstractSimulatedAnnealing,
    route::Elements.Route,
    new_route::Elements.Route,
    log_accept_prob::Float64,
)::Elements.Route =
    log_accept_prob > 0 || rand() < exp(log_accept_prob) ? new_route : route

stop_criteria(sa::AbstractSimulatedAnnealing, k::Int)::Bool =
    get_early_stop(sa) && k > get_stop_after(sa)

function run_sa(sa::AbstractSimulatedAnnealing)::Elements.Route
    route = initial_route(sa)
    route_value = compute_value(sa, route)
    best_route = route
    best_value = route_value
    for k ∈ 1:get_n_iter(sa)
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
