# MARK: Reversion Kernel
"""
Kernel that performs a reversion of a segment of the route.

E.g.: For a route ``[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]``, if ``i=3`` and ``j=7``
are selected as indices to reverse the route, the new route would be
``[1, 2, 7, 6, 5, 4, 3, 8, 9, 10]``.
"""
struct ReversionKernel <: AbstractKernel
    seed::Union{Int, Nothing}
    rng::Random.AbstractRNG

    ReversionKernel(rng::Random.AbstractRNG) = new(nothing, rng)
    ReversionKernel(seed::Int) = new(seed, Random.Xoshiro(seed))
    ReversionKernel() = new(nothing, Random.GLOBAL_RNG)
end

function _sample_reversion(route::Elements.Route, i::Int, j::Int)::Elements.Route
    i, j = min(i, j), max(i, j)
    new_route = copy(route)
    new_route[i:j] = reverse(new_route[i:j])
    return new_route
end

function sample(kernel::ReversionKernel, route::Elements.Route)::Elements.Route
    i, j = choice_without_replacement(kernel.rng, length(route), 2)
    return _sample_reversion(route, i, j)
end
