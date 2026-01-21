"""
    SimulatedAnnealingTSP(cities, kernel; n_iter=1_000_000, cooling_schedule=exponential_cooling_schedule(), early_stop=true, stop_after=10_000)

Main structure for the Simulated Annealing algorithm applied to the Traveling Salesman Problem (TSP).

# Arguments

  - `cities::Cities`: The set of cities to visit.
  - `kernel::K`: The kernel used to propose new routes.
  - `n_iter::Int`: Maximum number of iterations.
  - `cooling_schedule::Function`: Function that returns the temperature at iteration `k`.
  - `early_stop::Bool`: Whether to stop early if no improvement is found.
  - `stop_after::Int`: Number of iterations without improvement to trigger early stop.

# Example

```jldoctest
julia> using TspJulia.Elements, TspJulia.Kernels, TspJulia.SimulatedAnnealing

julia> cities = Cities([Point(0, 0), Point(1, 1), Point(0, 1)]);

julia> kernel = SwapKernel(42);

julia> sa = SimulatedAnnealingTSP(cities, kernel, n_iter = 100);

julia> typeof(sa)
SimulatedAnnealingTSP{SwapKernel}
```
"""
struct SimulatedAnnealingTSP{K <: Kernels.AbstractKernel} <: AbstractSimulatedAnnealing
    cities::Elements.Cities
    kernel::K
    n_iter::Int
    temperature::Any
    early_stop::Bool
    stop_after::Int

    function SimulatedAnnealingTSP(
        cities::Elements.Cities,
        kernel::K;
        n_iter::Int = 1_000_000,
        temperature::Any = ScheduleFns.exponential_cooling_schedule(
            T₀ = 100,
            ρ = 0.99,
        ),
        early_stop::Bool = true,
        stop_after::Int = 10_000,
    ) where {K <: Kernels.AbstractKernel}
        return new{K}(cities, kernel, n_iter, temperature, early_stop, stop_after)
    end
end

get_cities(sa::SimulatedAnnealingTSP)::Elements.Cities = sa.cities
get_kernel(sa::SimulatedAnnealingTSP)::Kernels.AbstractKernel = sa.kernel
get_n_iter(sa::SimulatedAnnealingTSP)::Int = sa.n_iter
get_temperature(sa::SimulatedAnnealingTSP)::Any = sa.temperature
get_early_stop(sa::SimulatedAnnealingTSP)::Bool = sa.early_stop
get_stop_after(sa::SimulatedAnnealingTSP)::Int = sa.stop_after
