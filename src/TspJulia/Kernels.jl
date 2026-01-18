module Kernels

using Random
using ..Elements

export AbstractKernel,
    SwapKernel,
    ReversionKernel,
    InsertionKernel,
    RandomWalkKernel,
    MixingKernel,
    sample,
    _sample_swap,
    _sample_reversion,
    _sample_insertion,
    _sample_mixing

# MARK: Abstract Kernel
"""
Abstract type for kernels.
"""
abstract type AbstractKernel end

function get_seed(kernel::AbstractKernel)::Union{Int, Nothing}
    if hasfield(typeof(kernel), :seed)
        return kernel.seed
    end
    throw(NotImplementedError("Kernel $(typeof(kernel)) has no field seed"))
end

function get_rng(kernel::AbstractKernel)::Random.AbstractRNG
    if hasfield(typeof(kernel), :rng)
        return kernel.rng
    end
    throw(NotImplementedError("Kernel $(typeof(kernel)) has no field rng"))
end

sample(kernel::AbstractKernel, route::Elements.Route)::Elements.Route =
    throw(NotImplementedError("sample not implemented for $(typeof(kernel))"))

function Base.show(io::IO, kernel::AbstractKernel)
    name = nameof(typeof(kernel))
    if get_seed(kernel) !== nothing
        print(io, "$name(seed=$(get_seed(kernel)))")
    elseif get_rng(kernel) !== Random.GLOBAL_RNG
        print(io, "$name(rng=$(get_rng(kernel)))")
    else
        print(io, "$name()")
    end
end

# MARK: Swap Kernel
"""
Kernel that performs a swap of two cities in the route.

E.g.: For a route ``[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]``, if ``i=3`` and ``j=7``
are selected as indices to swap, the new route would be
``[1, 2, 7, 4, 5, 6, 3, 8, 9, 10]``.
"""
struct SwapKernel <: AbstractKernel
    seed::Union{Int, Nothing}
    rng::Random.AbstractRNG

    SwapKernel(rng::Random.AbstractRNG) = new(nothing, rng)
    SwapKernel(seed::Int) = new(seed, Random.Xoshiro(seed))
    SwapKernel() = new(nothing, Random.GLOBAL_RNG)
end

function choice_without_replacement(
    rng::Random.AbstractRNG,
    n::Int,
    k::Int,
)::Vector{Int}
    return shuffle(rng, 1:n)[1:k]
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

# MARK: Insertion Kernel
"""
Kernel that performs an insertion of a city in the route.

E.g.: For a route ``[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]``, if ``i=3`` and ``j=7``
are selected as indices to insert the city 3 into the place 7,
the new route would be ``[1, 2, 4, 5, 6, 7, 3, 8, 9, 10]``.
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

# MARK: Mixing Kernel
"""
Kernel that mixes multiple kernels with given probabilities.

E.g.: For kernels ``K_1``, ``K_2`` and ``K_3`` with probabilities ``p_1``, ``p_2`` and ``p_3``,
the resulting new kernel would be ``K = p_1 K_1 + p_2 K_2 + p_3 K_3``.
"""
struct MixingKernel <: AbstractKernel
    kernels::Vector{AbstractKernel}
    probs::Vector{Float64}
    seed::Union{Int, Nothing}
    rng::Random.AbstractRNG

    function MixingKernel(
        kernels::AbstractVector{<:AbstractKernel},
        probs::AbstractVector{<:Real},
        seed::Union{Int, Nothing} = nothing,
        rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    )
        if length(kernels) != length(probs)
            throw(ArgumentError("The lengths of 'kernels' and 'probs' must be equals."))
        end
        probs = Float64.(probs)
        probs = Float64.(probs) .+ eps(Float64) * length(probs)
        probs = probs / sum(probs)
        indices = sortperm(probs; rev = true)
        return new(collect(kernels)[indices], probs[indices], seed, rng)
    end

    MixingKernel(
        kernels::AbstractVector{<:AbstractKernel},
        probs::AbstractVector{<:Real},
        seed::Int,
    ) = new(kernels, probs, seed, Random.Xoshiro(seed))

    MixingKernel(
        kernels::AbstractVector{<:AbstractKernel},
        probs::AbstractVector{<:Real},
        rng::Random.AbstractRNG,
    ) = new(kernels, probs, nothing, rng)
end

function Base.show(io::IO, kernel::MixingKernel)
    repr =
        "MixingKernelTSP(" *
        join(
            [
                "$(kernel.kernels[i]): prob=$(round(kernel.probs[i], digits=4))" for
                i ∈ eachindex(kernel.kernels)
            ],
            ", ",
        ) *
        ")"
    return print(io, repr)
end

function _choose_kernel(kernel::MixingKernel, u::Real)::AbstractKernel
    cumsum_p = cumsum(kernel.probs)
    for i ∈ eachindex(cumsum_p)
        if u < cumsum_p[i]
            return kernel.kernels[i]
        end
    end
    return kernel.kernels[end]
end

function _sample_mixing(
    kernel::MixingKernel,
    route::Elements.Route,
    u::Real,
)::Elements.Route
    kernel = _choose_kernel(kernel, u)
    return sample(kernel, route)
end

function sample(kernel::MixingKernel, route::Elements.Route)::Elements.Route
    u = rand(kernel.rng)
    return _sample_mixing(kernel, route, u)
end

end  # module Kernels
