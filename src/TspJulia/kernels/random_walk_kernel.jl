# MARK: Random Walk Kernel
"""
    RandomWalkKernel([seed|rng])

Kernel that performs a random walk over possible path combinations.
Returns a different path than the one initially given.

# Example

```jldoctest
julia> using TspJulia.Elements, TspJulia.Kernels

julia> route = Route(collect(1:5));

julia> kernel = RandomWalkKernel(42);

julia> new_route = sample(kernel, route);

julia> new_route == Route([1, 3, 4, 2, 5])
true
```
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
