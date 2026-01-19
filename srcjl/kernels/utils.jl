function choice_without_replacement(
    rng::Random.AbstractRNG,
    n::Int,
    k::Int,
)::Vector{Int}
    return shuffle(rng, 1:n)[1:k]
end
