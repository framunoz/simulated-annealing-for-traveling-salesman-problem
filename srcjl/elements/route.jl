"""
Route struct.
"""
struct Route
    route::Vector{Int16}
end

Route(route::AbstractVector{Integer}) = Route(Vector{Int16}(route))

Base.copy(route::Route)::Route = Route(copy(route.route))
Base.isequal(route1::Route, route2::Route)::Bool = isequal(route1.route, route2.route)
Base.length(route::Route)::Int = length(route.route)
Base.getindex(route::Route, i)::Int16 = route.route[i]
Base.setindex!(route::Route, v, i) = route.route[i] = v
Base.getindex(route::Route, i::UnitRange)::Vector{Int16} = route.route[i]
Base.setindex!(route::Route, v::Vector, i::UnitRange) = route.route[i] = v
Base.lastindex(route::Route) = lastindex(route.route)
Base.show(io::IO, route::Route) = print(io, "Route(" * join(route.route, ", ") * ")")
Base.:(==)(route1::Route, route2::Route)::Bool = route1.route == route2.route
