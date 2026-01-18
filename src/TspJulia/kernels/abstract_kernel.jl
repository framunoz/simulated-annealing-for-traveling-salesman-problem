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
