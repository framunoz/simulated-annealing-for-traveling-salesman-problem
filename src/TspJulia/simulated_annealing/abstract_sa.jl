abstract type AbstractSimulatedAnnealing end

get_cities(sa::AbstractSimulatedAnnealing)::Elements.Cities =
    throw(NotImplementedError("get_cities not implemented for $(typeof(sa))"))
get_kernel(sa::AbstractSimulatedAnnealing)::Elements.AbstractKernel =
    throw(NotImplementedError("get_kernel not implemented for $(typeof(sa))"))
get_n_iter(sa::AbstractSimulatedAnnealing)::Int =
    throw(NotImplementedError("get_n_iter not implemented for $(typeof(sa))"))
get_temperature(sa::AbstractSimulatedAnnealing)::Function =
    throw(NotImplementedError("get_cooling_schedule not implemented for $(typeof(sa))"))
get_early_stop(sa::AbstractSimulatedAnnealing)::Bool =
    throw(NotImplementedError("get_early_stop not implemented for $(typeof(sa))"))
get_stop_after(sa::AbstractSimulatedAnnealing)::Int =
    throw(NotImplementedError("get_stop_after not implemented for $(typeof(sa))"))

callback_after_iteration(sa::AbstractSimulatedAnnealing, k::Int) = nothing
callback_after_acceptance(sa::AbstractSimulatedAnnealing, k::Int) = nothing
