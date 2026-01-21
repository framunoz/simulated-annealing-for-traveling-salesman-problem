
import typing as t
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ..utils import distance_matrix

__all__ = ["Point", "Cities", "Route"]

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Route:
    route: list[int]

    def __len__(self) -> int:
        return len(self.route)

    def __getitem__(self, item) -> int:
        return self.route[item]
    
    def __iter__(self):
        return iter(self.route)
        
    def copy(self) -> "Route":
        return Route(list(self.route))

    def to_jl(self):
        # Helper for compatibility if needed, though this is pure python impl
        return self.route

@dataclass
class Cities:
    """
    Wrapper for points. Distance matrix is computed lazily.
    """
    points: list[Point]
    _dist_matrix: t.Optional[np.ndarray] = field(default=None, repr=False, init=False)

    @cached_property
    def dist_matrix(self) -> np.ndarray:
        # Convert points to array Nx2
        coords = np.array([[p.x, p.y] for p in self.points])
        return distance_matrix(coords)

    def __len__(self) -> int:
        return len(self.points)

    def total_distance(self, route: Route) -> float:
        """Compute the total distance of the route."""
        # Logic from original sa.py
        r = route.route
        rot_route = deque(r)
        rot_route.rotate(-1)
        
        matrix = self.dist_matrix
        return sum([matrix[i, j] for i, j in zip(r, rot_route)])
