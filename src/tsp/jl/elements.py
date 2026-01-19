"""Python wrappers for Julia Elements module (Point, Cities, Route)."""

import typing as t
from dataclasses import dataclass

from juliacall import AnyValue
from juliacall import Main as jl

from tsp.jl.common import JuliaWrapper

__all__ = ["Point", "Cities", "Route"]

# Initialize Julia modules
jl.include("src/TspJulia/elements/Elements.jl")
jl.seval("using .Elements")


@dataclass
class Point(JuliaWrapper):
    """Python wrapper for Julia Point type.
    
    Attributes:
        x: X coordinate
        y: Y coordinate
    """
    x: float
    y: float

    def _to_jl_impl(self) -> AnyValue:
        """Convert to Julia Point."""
        return jl.Point(self.x, self.y)

    @classmethod
    def from_jl(cls, jl_value: AnyValue) -> t.Self:
        """Create Point from Julia Point."""
        return cls(x=float(jl_value.x), y=float(jl_value.y))


@dataclass
class Cities(JuliaWrapper):
    """Python wrapper for Julia Cities type.
    
    Attributes:
        cities: List of Point objects
    """
    cities: list[Point]

    def _to_jl_impl(self) -> AnyValue:
        """Convert to Julia Cities."""
        jl_points = [city.to_jl() for city in self.cities]
        return jl.Cities(jl_points)

    @classmethod
    def from_jl(cls, jl_value: AnyValue) -> t.Self:
        """Create Cities from Julia Cities."""
        cities = [Point.from_jl(jl_value[i]) for i in range(1, len(jl_value) + 1)]
        return cls(cities=cities)

    def __len__(self) -> int:
        """Return number of cities."""
        return len(self.cities)

    def __getitem__(self, index: int) -> Point:
        """Get city at index (0-indexed)."""
        return self.cities[index]


@dataclass
class Route(JuliaWrapper):
    """Python wrapper for Julia Route type.
    
    Attributes:
        route: List of city indices (0-indexed in Python, 1-indexed in Julia)
    """
    route: list[int]

    def _to_jl_impl(self) -> AnyValue:
        """Convert to Julia Route (converting to 1-indexed)."""
        # Julia uses 1-indexed arrays
        jl_route = [i + 1 for i in self.route]
        return jl.Route(jl_route)

    @classmethod
    def from_jl(cls, jl_value: AnyValue) -> t.Self:
        """Create Route from Julia Route (converting to 0-indexed)."""
        # Convert from 1-indexed to 0-indexed
        route = [int(jl_value[i]) - 1 for i in range(1, len(jl_value) + 1)]
        return cls(route=route)

    def __len__(self) -> int:
        """Return length of route."""
        return len(self.route)

    def __getitem__(self, index: int) -> int:
        """Get city index at position (0-indexed)."""
        return self.route[index]
