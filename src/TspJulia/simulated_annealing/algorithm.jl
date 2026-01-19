function initial_route(sa::AbstractSimulatedAnnealing)::Tuple{Elements.Route, Float64}
    route = Elements.Route(1:length(get_cities(sa)))
    route_value = compute_value(sa, route)
    return route, route_value
end

function sample_from_kernel(
    sa::AbstractSimulatedAnnealing,
    route::Elements.Route,
)::Tuple{Elements.Route, Float64}
    new_route = get_kernel(sa)(route)
    dist_new_route = compute_value(sa, new_route)
    return new_route, dist_new_route
end

function compute_value(sa::AbstractSimulatedAnnealing, route::Elements.Route)::Float64
    return total_distance(get_cities(sa), route)
end

function compute_log_acceptance_prob(
    sa::AbstractSimulatedAnnealing,
    route::Elements.Route,
    new_route::Elements.Route,
    k::Int,
)::Float64
    gap = compute_value(sa, new_route) - compute_value(sa, route)
    if gap <= 0
        return 0.0
    end
    return -gap / (get_temperature(sa)(k) + eps(Float64))
end

function stop_criteria(sa::AbstractSimulatedAnnealing, stop_counter::Int)::Bool
    return get_early_stop(sa) && stop_counter > get_stop_after(sa)
end

function run_sa(sa::AbstractSimulatedAnnealing)::Tuple{Elements.Route, Float64}
    route, route_value = initial_route(sa)

    best_route = route
    best_value = route_value

    stop_counter = 0
    for k ∈ 1:get_n_iter(sa)
        new_route, new_route_value = sample_from_kernel(sa, route)

        log_accept_prob = compute_log_acceptance_prob(sa, route, new_route, k)

        if log_accept_prob > 0 || rand() < exp(log_accept_prob)
            stop_counter = 0

            if new_route_value < best_value
                best_route = new_route
                best_value = new_route_value
            end

            route = new_route
            route_value = new_route_value

            callback_after_acceptance(sa, k)
        else
            stop_counter += 1
        end

        callback_after_iteration(sa, k)

        if stop_criteria(sa, stop_counter)
            break
        end
    end
    return best_route, best_value
end
