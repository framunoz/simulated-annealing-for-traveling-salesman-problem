module Elements

export Point, Cities, Route, distance, compute_distance_matrix, total_distance

"""
Point struct.
"""
struct Point
    x::Float32
    y::Float32
end

Point(x::Real, y::Real) = Point(Float32(x), Float32(y))

"""
Cities struct.
"""
struct Cities
    cities::Vector{Point}
end

Cities(cities::AbstractVector{Point}) = Cities(collect(cities))

Base.copy(cities::Cities)::Cities = Cities(copy(cities.cities))
Base.isequal(cities1::Cities, cities2::Cities)::Bool = isequal(cities1.cities, cities2.cities)
Base.length(cities::Cities)::Int = length(cities.cities)
Base.getindex(cities::Cities, i)::Point = cities.cities[i]
Base.setindex!(cities::Cities, i, v::Point) = cities.cities[i] = v
Base.getindex(cities::Cities, i::UnitRange)::Vector{Point} = cities.cities[i]
Base.setindex!(cities::Cities, i::UnitRange, v::Vector{Point}) = cities.cities[i] = v
Base.show(io::IO, cities::Cities) = print(io, "Cities(" * join(cities.cities, ", ") * ")")

"""
Route struct.
"""
struct Route
    route::Vector{Int16}
end

Route(route::AbstractVector) = Route(Vector{Int16}(route))

Base.copy(route::Route)::Route = Route(copy(route.route))
Base.isequal(route1::Route, route2::Route)::Bool = isequal(route1.route, route2.route)
Base.length(route::Route)::Int = length(route.route)
Base.getindex(route::Route, i)::Int16 = route.route[i]
Base.setindex!(route::Route, i, v) = route.route[i] = v
Base.getindex(route::Route, i::UnitRange)::Vector{Int16} = route.route[i]
Base.setindex!(route::Route, i::UnitRange, v::Vector) = route.route[i] = v
Base.lastindex(route::Route) = lastindex(route.route)
Base.show(io::IO, route::Route) = print(io, "Route(" * join(route.route, ", ") * ")")

"""
Distance function.
"""
function distance(p1::Point, p2::Point)
    return sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
end

"""
Compute distance matrix.
"""
function compute_distance_matrix(cities::Cities)
    return [
        distance(cities.cities[i], cities.cities[j])
        for i in 1:length(cities.cities), j in 1:length(cities.cities)
    ]
end

function total_distance(cities::Cities, route::Route)
    dist_matrix = compute_distance_matrix(cities)
    return sum(
        dist_matrix[route[i], route[i+1]] for i in 1:length(route)-1
    ) + dist_matrix[route[end], route[1]]
end

end  # module Elements