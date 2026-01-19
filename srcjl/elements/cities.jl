"""
Cities struct.
"""
struct Cities
    cities::Vector{Point}
end

Cities(cities::AbstractVector{Point}) = Cities(collect(cities))

Base.copy(cities::Cities)::Cities = Cities(copy(cities.cities))
Base.isequal(cities1::Cities, cities2::Cities)::Bool =
    isequal(cities1.cities, cities2.cities)
Base.length(cities::Cities)::Int = length(cities.cities)
Base.getindex(cities::Cities, i)::Point = cities.cities[i]
Base.setindex!(cities::Cities, v::Point, i) = cities.cities[i] = v
Base.getindex(cities::Cities, i::UnitRange)::Vector{Point} = cities.cities[i]
Base.setindex!(cities::Cities, v::Vector{Point}, i::UnitRange) = cities.cities[i] = v

function Base.show(io::IO, cities::Cities)
    if length(cities.cities) == 0
        return print(io, "Cities()")
    end
    cities_repr = ""
    for (i, city) ∈ enumerate(cities.cities)
        cities_repr *= "$(i):$(city), "
    end
    repr = "Cities($(cities_repr[1:end-2]))"
    return print(io, repr)
end

function Base.getindex(cities::Cities, route::Route)::Cities
    return Cities(cities.cities[route.route])
end