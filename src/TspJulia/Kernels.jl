module Kernels

using Random
using ..Elements

export AbstractKernel,
    SwapKernelTSP,
    ReversionKernelTSP,
    InsertionKernelTSP,
    RandomWalkKernelTSP,
    MixingKernelTSP,
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

sample(kernel::AbstractKernel, route::Route)::Route =
    throw(NotImplementedError("sample not implemented for $(typeof(kernel))"))

function Base.show(io::IO, kernel::AbstractKernel)
    name = nameof(typeof(kernel))
    if hasfield(typeof(kernel), :seed) && kernel.seed !== nothing
        print(io, "$name(seed=$(kernel.seed))")
    elseif hasfield(typeof(kernel), :rng) && kernel.rng !== Random.GLOBAL_RNG
        print(io, "$name(rng=$(kernel.rng))")
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
struct SwapKernelTSP <: AbstractKernel
    seed::Union{Int,Nothing}
    rng::Random.AbstractRNG

    SwapKernelTSP(rng::Random.AbstractRNG) = new(nothing, rng)
    SwapKernelTSP(seed::Int) = new(seed, Random.Xoshiro(seed))
    SwapKernelTSP() = new(nothing, Random.GLOBAL_RNG)
end


function choice_without_replacement(rng::Random.AbstractRNG, n::Int, k::Int)::Vector{Int}
    return shuffle(rng, 1:n)[1:k]
end


function _sample_swap(route::Route, i, j)::Route
    new_route = copy(route)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route
end


function sample(kernel::SwapKernelTSP, route::Route)::Route
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
struct ReversionKernelTSP <: AbstractKernel
    seed::Union{Int,Nothing}
    rng::Random.AbstractRNG

    ReversionKernelTSP(rng::Random.AbstractRNG) = new(nothing, rng)
    ReversionKernelTSP(seed::Int) = new(seed, Random.Xoshiro(seed))
    ReversionKernelTSP() = new(nothing, Random.GLOBAL_RNG)
end

function _sample_reversion(route::Route, i, j)::Route
    i, j = min(i, j), max(i, j)
    new_route = copy(route)
    new_route[i:j] = reverse(new_route[i:j])
    return new_route
end

function sample(kernel::ReversionKernelTSP, route::Route)::Route
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
struct InsertionKernelTSP <: AbstractKernel
    seed::Union{Int,Nothing}
    rng::Random.AbstractRNG

    InsertionKernelTSP(rng::Random.AbstractRNG) = new(nothing, rng)
    InsertionKernelTSP(seed::Int) = new(seed, Random.Xoshiro(seed))
    InsertionKernelTSP() = new(nothing, Random.GLOBAL_RNG)
end

function _sample_insertion(route::Route, i, j)::Route
    new_route = copy(route)
    elem_i = new_route[i]
    new_route = Route(vcat(new_route[1:i-1], new_route[i+1:j], elem_i, new_route[j+1:end]))
    return new_route
end


function sample(kernel::InsertionKernelTSP, route::Route)::Route
    i, j = choice_without_replacement(kernel.rng, length(route), 2)
    return _sample_insertion(route, i, j)
end


# MARK: Random Walk Kernel
"""
Kernel that performs a random walk over possible path combinations.
Returns a different path than the one initially given.
"""
struct RandomWalkKernelTSP <: AbstractKernel
    seed::Union{Int,Nothing}
    rng::Random.AbstractRNG

    RandomWalkKernelTSP(rng::Random.AbstractRNG) = new(nothing, rng)
    RandomWalkKernelTSP(seed::Int) = new(seed, Random.Xoshiro(seed))
    RandomWalkKernelTSP() = new(nothing, Random.GLOBAL_RNG)
end

function sample(kernel::RandomWalkKernelTSP, route::Route)::Route
    new_route = route
    while new_route == route
        # Concatenate the first element of the route with the shuffled rest of the route
        new_route = Route(vcat(route[1], shuffle(kernel.rng, route[2:end])))
    end
    return new_route
end


# MARK: Mixing Kernel
"""
Kernel that mixes multiple kernels with given probabilities.

E.g.: For kernels ``K_1``, ``K_2`` and ``K_3`` with probabilities ``p_1``, ``p_2`` and ``p_3``,
the resulting new kernel would be ``K = p_1 K_1 + p_2 K_2 + p_3 K_3``.
"""
struct MixingKernelTSP <: AbstractKernel
    kernels::Vector{AbstractKernel}
    probs::Vector{Float64}
    seed::Union{Int,Nothing}
    rng::Random.AbstractRNG

    function MixingKernelTSP(
        kernels::AbstractVector{<:AbstractKernel},
        probs::AbstractVector{<:Real},
        seed::Union{Int,Nothing}=nothing,
        rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    )
        if length(kernels) != length(probs)
            throw(ArgumentError("The lengths of 'kernels' and 'probs' must be equals."))
        end
        probs = Float64.(probs)
        probs = Float64.(probs) .+ eps(Float64) * length(probs)
        probs = probs / sum(probs)
        indices = sortperm(probs, rev=true)
        return new(collect(kernels)[indices], probs[indices], seed, rng)
    end

    MixingKernelTSP(
        kernels::AbstractVector{<:AbstractKernel},
        probs::AbstractVector{<:Real},
        seed::Int,
    ) = new(kernels, probs, seed, Random.Xoshiro(seed))

    MixingKernelTSP(
        kernels::AbstractVector{<:AbstractKernel},
        probs::AbstractVector{<:Real},
        rng::Random.AbstractRNG,
    ) = new(kernels, probs, nothing, rng)

end

function Base.show(io::IO, kernel::MixingKernelTSP)
    repr =
        "MixingKernelTSP(" *
        join(
            [
                "$(kernel.kernels[i]): prob=$(round(kernel.probs[i], digits=4))" for
                i in eachindex(kernel.kernels)
            ],
            ", ",
        ) *
        ")"
    print(io, repr)
end

function _sample_mixing(kernel::MixingKernelTSP, route::Route, u::Float64)::Route
    cumsum_p = cumsum(kernel.probs)
    for i in eachindex(cumsum_p)
        if u < cumsum_p[i]
            return sample(kernel.kernels[i], route)
        end
    end
    return sample(kernel.kernels[end], route)
end

function sample(kernel::MixingKernelTSP, route::Route)::Route
    u = rand(kernel.rng)
    return _sample_mixing(kernel, route, u)
end

end  # module Kernels
