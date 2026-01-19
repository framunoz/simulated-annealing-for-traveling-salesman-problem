# MARK: Insertion Kernel
"""
    InsertionKernel([seed|rng])

Kernel that performs an insertion of a city in the route.

E.g.: For a route ``[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]``, if ``i=3`` and ``j=7``
are selected as indices to insert the city 3 into the place 7,
the new route would be ``[1, 2, 4, 5, 6, 7, 3, 8, 9, 10]``.

# Example

```jldoctest
julia> using TspJulia.Elements, TspJulia.Kernels

julia> route = Route(collect(1:10));

julia> kernel = InsertionKernel(42);

julia> new_route = sample(kernel, route);

julia> new_route == Route([1, 3, 2, 4, 5, 6, 7, 8, 9, 10])
true
```
"""
struct InsertionKernel <: AbstractKernel
    seed::Union{Int, Nothing}
    rng::Random.AbstractRNG

    InsertionKernel(rng::Random.AbstractRNG) = new(nothing, rng)
    InsertionKernel(seed::Int) = new(seed, Random.Xoshiro(seed))
    InsertionKernel() = new(nothing, Random.GLOBAL_RNG)
end

function _sample_insertion(route::Elements.Route, i::Int, j::Int)::Elements.Route
    if i == j
        return copy(route)
    end
    data = copy(route.route)
    elem_i = data[i]
    deleteat!(data, i)
    insert!(data, j, elem_i)
    return Elements.Route(data)
end

function sample(kernel::InsertionKernel, route::Elements.Route)::Elements.Route
    i, j = choice_without_replacement(kernel.rng, length(route), 2)
    return _sample_insertion(route, i, j)
end
