# MARK: Random Walk Kernel
"""
Kernel that performs a random walk over possible path combinations.
Returns a different path than the one initially given.
"""
struct RandomWalkKernel <: AbstractKernel
    seed::Union{Int, Nothing}
    rng::Random.AbstractRNG

    RandomWalkKernel(rng::Random.AbstractRNG) = new(nothing, rng)
    RandomWalkKernel(seed::Int) = new(seed, Random.Xoshiro(seed))
    RandomWalkKernel() = new(nothing, Random.GLOBAL_RNG)
end

function sample(kernel::RandomWalkKernel, route::Elements.Route)::Elements.Route
    new_route = route
    while new_route == route
        # Concatenate the first element of the route with the shuffled rest of the route
        new_route = Elements.Route(vcat(route[1], shuffle(kernel.rng, route[2:end])))
    end
    return new_route
end
