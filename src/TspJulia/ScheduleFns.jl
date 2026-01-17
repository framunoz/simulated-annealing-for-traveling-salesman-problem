module ScheduleFns

export exponential_cooling_schedule, log_cooling_schedule, linear_cooling_schedule, fast_cooling_schedule, slow_cooling_schedule, linear_decrease_cooling_schedule

function exponential_cooling_schedule(T_0::Real, rho::Real)
    function aux(k::Int)
        return T_0 * rho^k
    end
    return aux
end


function log_cooling_schedule(T_0::Real, k_0::Real)
    function aux(k::Int)
        return T_0 * log(k_0) / log(k + k_0)
    end
    return aux
end


function linear_cooling_schedule(T_0::Real, T_f::Real, n_iter::Int)
    function aux(k::Int)
        return T_0 + (T_f - T_0) * k / n_iter
    end
    return aux
end


function fast_cooling_schedule(T_0::Real, rho::Real, n_iter::Int)
    function aux(k::Int)
        return T_0 * rho^(k^2 / n_iter)
    end
    return aux
end


function slow_cooling_schedule(T_0::Real, rho::Real, n_iter::Int)
    function aux(k::Int)
        return T_0 * rho^(sqrt(k) / n_iter)
    end
    return aux
end


function linear_decrease_cooling_schedule(T_0::Real, T_f::Real, n_iter::Int)
    function aux(k::Int)
        return T_0 - (T_0 - T_f) * k / n_iter
    end
    return aux
end

end
