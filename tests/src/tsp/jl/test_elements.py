
import pytest
from juliacall import AnyValue  # type: ignore

from tsp.jl.elements import Cities, Point, Route


def test_point_roundtrip():
    p = Point(1.5, 2.5)
    jl_p = p.to_jl()
    
    assert str(type(jl_p)) == "<class 'juliacall.AnyValue'>"
    # Julia properties access
    assert jl_p.x == 1.5
    assert jl_p.y == 2.5
    
    p_back = Point.from_jl(jl_p)
    assert p_back == p
    assert p_back.x == 1.5
    assert p_back.y == 2.5

def test_cities_roundtrip():
    points = [Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 0.0)]
    cities = Cities(points)
    
    assert len(cities) == 3
    assert cities[1] == points[1]
    
    jl_cities = cities.to_jl()
    
    cities_back = Cities.from_jl(jl_cities)
    assert len(cities_back) == 3
    assert cities_back.cities == points
    assert cities_back == cities

def test_route_indexing_conversion():
    # Python 0-indexed: [0, 2, 1]
    # Julia 1-indexed:  [1, 3, 2]
    route_indices = [0, 2, 1]
    route = Route(route_indices)
    
    assert len(route) == 3
    assert route[1] == 2
    
    jl_route = route.to_jl()
    
    # Verify Julia side values (1-based)
    # indexed by 1 in Julia
    assert jl_route[1] == 1
    assert jl_route[2] == 3
    assert jl_route[3] == 2
    
    route_back = Route.from_jl(jl_route)
    assert route_back.route == route_indices
    assert route_back == route
