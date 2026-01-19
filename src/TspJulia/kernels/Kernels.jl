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

include("utils.jl")
include("abstract_kernel.jl")
include("swap_kernel.jl")
include("reversion_kernel.jl")
include("insertion_kernel.jl")
include("random_walk_kernel.jl")
include("mixing_kernel.jl")

end  # module Kernels
