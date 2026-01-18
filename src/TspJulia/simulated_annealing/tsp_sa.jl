struct SimulatedAnnealingTSP{K <: Kernels.AbstractKernel} <: AbstractSimulatedAnnealing
    cities::Elements.Cities
    kernel::K
    n_iter::Int
    cooling_schedule::Function
    early_stop::Bool
    stop_after::Int

    function SimulatedAnnealingTSP(
        cities::Elements.Cities,
        kernel::K,
        n_iter::Int = 1_000_000,
        cooling_schedule::Function = ScheduleFns.exponential_cooling_schedule(
            T₀ = 100,
            ρ = 0.99,
        ),
        early_stop::Bool = true,
        stop_after::Int = 10_000,
    ) where {K <: Kernels.AbstractKernel}
        return new{K}(cities, kernel, n_iter, cooling_schedule, early_stop, stop_after)
    end
end

get_cities(sa::SimulatedAnnealingTSP)::Elements.Cities = sa.cities
get_kernel(sa::SimulatedAnnealingTSP)::Kernels.AbstractKernel = sa.kernel
get_n_iter(sa::SimulatedAnnealingTSP)::Int = sa.n_iter
get_cooling_schedule(sa::SimulatedAnnealingTSP)::Function = sa.cooling_schedule
get_early_stop(sa::SimulatedAnnealingTSP)::Bool = sa.early_stop
get_stop_after(sa::SimulatedAnnealingTSP)::Int = sa.stop_after
