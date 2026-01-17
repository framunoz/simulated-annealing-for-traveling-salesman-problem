module SimulatedAnnealing

export SimulatedAnnealingTSP, run_sa

using Random
using ..Elements
using ..Kernels
using ..ScheduleFns

struct SimulatedAnnealingTSP{K<:AbstractKernel}
    cities::Cities
    kernel::K
    n_iter::Int
    cooling_schedule::Function
    early_stop::Bool
    stop_after::Int
    seed::Union{Int,Random.AbstractRNG,Nothing}
end

function SimulatedAnnealingTSP(
    cities::Cities,
    kernel::K,
    n_iter::Int=1_000_000,
    cooling_schedule::Function=exponential_cooling_schedule(100, 0.99),
    early_stop::Bool=true,
    stop_after::Int=10_000,
    seed::Union{Int,Random.AbstractRNG,Nothing}=nothing
) where {K<:AbstractKernel}
    return SimulatedAnnealingTSP{K}(cities, kernel, n_iter, cooling_schedule, early_stop, stop_after, seed)
end


function step_sa(sa::SimulatedAnnealingTSP, route::Route, k::Int)

    new_route = sample(sa.kernel, route)

    gap = total_distance(sa.cities, new_route) - total_distance(sa.cities, route)

    log_accept_prob = -gap / sa.cooling_schedule(k)

    if log_accept_prob > 0 || rand() < exp(log_accept_prob)
        return new_route
    else
        return route
    end

end


function run_sa(sa::SimulatedAnnealingTSP)

    route = Route(1:length(sa.cities))

    for k in 1:sa.n_iter
        route = step_sa(sa, route, k)

        if sa.early_stop && k > sa.stop_after
            println(k)
            println(route)
            println(total_distance(sa.cities, route))
            println("\n")
            break
        end
    end

    return route
end


end  # module SimulatedAnnealing