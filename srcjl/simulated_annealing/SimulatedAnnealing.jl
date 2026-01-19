module SimulatedAnnealing

export AbstractSimulatedAnnealing, SimulatedAnnealingTSP, run_sa, SAStatsProxy

using Random
using ..Elements
using ..Kernels
using ..ScheduleFns

include("abstract_sa.jl")
include("algorithm.jl")
include("tsp_sa.jl")
include("stats_proxy.jl")

end  # module SimulatedAnnealing
