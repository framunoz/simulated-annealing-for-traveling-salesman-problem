# MARK: Swap Kernel
"""
    SwapKernel([seed|rng])

Kernel that performs a swap of two cities in the route.

E.g.: For a route ``[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]``, if ``i=3`` and ``j=7``
are selected as indices to swap, the new route would be
``[1, 2, 7, 4, 5, 6, 3, 8, 9, 10]``.

# Example

```jldoctest
julia> using TspJulia.Elements, TspJulia.Kernels

julia> route = Route(collect(1:10));

julia> kernel = SwapKernel(42);

julia> new_route = sample(kernel, route);

julia> new_route == Route([1, 3, 2, 4, 5, 6, 7, 8, 9, 10])
true
```
"""
struct SwapKernel <: AbstractKernel
    seed::Union{Int, Nothing}
    rng::Random.AbstractRNG

    SwapKernel(rng::Random.AbstractRNG) = new(nothing, rng)
    SwapKernel(seed::Int) = new(seed, Random.Xoshiro(seed))
    SwapKernel() = new(nothing, Random.GLOBAL_RNG)
end

function _sample_swap(route::Elements.Route, i::Int, j::Int)::Elements.Route
    new_route = copy(route)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route
end

function sample(kernel::SwapKernel, route::Elements.Route)::Elements.Route
    i, j = choice_without_replacement(kernel.rng, length(route), 2)
    return _sample_swap(route, i, j)
end
