"""
Point struct.
"""
struct Point
    x::Float32
    y::Float32
end

Point(x::Real, y::Real) = Point(Float32(x), Float32(y))

Base.show(io::IO, p::Point) = print(io, "P($(p.x), $(p.y))")
