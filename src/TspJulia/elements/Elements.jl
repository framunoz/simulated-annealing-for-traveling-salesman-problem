module Elements

include("point.jl")
include("cities.jl")
include("route.jl")
include("functions.jl")

export Point, Cities, Route, distance, compute_distance_matrix, total_distance

"""
Distance function.
"""
distance(p1::Point, p2::Point) = √((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

"""
Compute distance matrix.
"""
function compute_distance_matrix(cities::Cities)
    return [
        distance(cities.cities[i], cities.cities[j]) for i ∈ 1:length(cities.cities),
        j ∈ 1:length(cities.cities)
    ]
end

function total_distance(cities::Cities, route::Route)
    dist_matrix = compute_distance_matrix(cities)
    return sum(dist_matrix[route[i], route[i + 1]] for i ∈ 1:(length(route) - 1)) +
           dist_matrix[route[end], route[1]]
end

end  # module Elements
