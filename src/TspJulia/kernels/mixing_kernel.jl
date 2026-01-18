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
