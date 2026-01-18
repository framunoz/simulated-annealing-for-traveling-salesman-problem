module ScheduleFns

export exponential_cooling_schedule,
    log_cooling_schedule,
    linear_increase_cooling_schedule,
    fast_cooling_schedule,
    slow_cooling_schedule,
    linear_decrease_cooling_schedule


exponential_cooling_schedule(T₀::Real, ρ::Real) = k -> T₀ * ρ^k

log_cooling_schedule(T₀::Real, k₀::Real) = k -> T₀ * log(k₀) / log(k + k₀)

linear_increase_cooling_schedule(T₀::Real, T₁::Real, n_iter::Int) =
    k -> T₀ + (T₁ - T₀) * k / n_iter

linear_decrease_cooling_schedule(T₀::Real, T₁::Real, n_iter::Int) =
    k -> T₀ - (T₀ - T₁) * k / n_iter

fast_cooling_schedule(T₀::Real, ρ::Real, n_iter::Int) = k -> T₀ * ρ^(k^2 / n_iter)

slow_cooling_schedule(T₀::Real, ρ::Real, n_iter::Int) = k -> T₀ * ρ^(√k / n_iter)

end
