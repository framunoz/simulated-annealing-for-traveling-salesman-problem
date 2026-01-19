"""Python wrappers for Julia Kernels module."""

import abc
import typing as t
from dataclasses import dataclass

from juliacall import AnyValue  # type: ignore
from juliacall import Main as jl  # type: ignore

from .common import JuliaWrapper
from .elements import Route

__all__ = [
    "SwapKernel",
    "ReversionKernel",
    "InsertionKernel",
    "RandomWalkKernel",
    "MixingKernel",
]

# Initialize Julia modules
# Initialize Julia modules
if not jl.seval("isdefined(Main, :TspJulia)"):
    jl.include("srcjl/TspJulia.jl")
jl.seval("using .TspJulia")
jl.seval("using .TspJulia.Kernels")


@dataclass
class AbstractKernel(JuliaWrapper, abc.ABC):
    """Abstract base class for Julia Kernels."""

    @abc.abstractmethod
    def sample(self, route: Route) -> Route:
        """Sample a new route using the kernel."""
        ...

    def __call__(self, route: Route) -> Route:
        """Sample a new route using the kernel."""
        return self.sample(route)


@dataclass
class SwapKernel(AbstractKernel):
    """Python wrapper for Julia SwapKernel.
    
    Kernel that performs a swap of two cities in the route.
    
    Attributes:
        seed: Random seed (optional)
    """
    seed: int | None = None

    @t.override
    def _to_jl_impl(self) -> AnyValue:
        """Convert to Julia SwapKernel."""
        if self.seed is not None:
            return jl.SwapKernel(self.seed)
        return jl.SwapKernel()

    @classmethod
    @t.override
    def from_jl(cls, jl_value: AnyValue) -> t.Self:
        """Create SwapKernel from Julia SwapKernel."""
        seed = jl.TspJulia.Kernels.get_seed(jl_value)
        return cls(seed=int(seed) if seed is not None else None)

    @t.override
    def sample(self, route: Route) -> Route:
        """Sample a new route by swapping two cities."""
        jl_kernel = self.to_jl()
        jl_route = route.to_jl()
        jl_new_route = jl.sample(jl_kernel, jl_route)
        return Route.from_jl(jl_new_route)


@dataclass
class ReversionKernel(AbstractKernel):
    """Python wrapper for Julia ReversionKernel.
    
    Kernel that reverses a segment of the route.
    
    Attributes:
        seed: Random seed (optional)
    """
    seed: int | None = None

    @t.override
    def _to_jl_impl(self) -> AnyValue:
        """Convert to Julia ReversionKernel."""
        if self.seed is not None:
            return jl.ReversionKernel(self.seed)
        return jl.ReversionKernel()

    @classmethod
    @t.override
    def from_jl(cls, jl_value: AnyValue) -> t.Self:
        """Create ReversionKernel from Julia ReversionKernel."""
        seed = jl.TspJulia.Kernels.get_seed(jl_value)
        return cls(seed=int(seed) if seed is not None else None)

    @t.override
    def sample(self, route: Route) -> Route:
        """Sample a new route by reversing a segment."""
        jl_kernel = self.to_jl()
        jl_route = route.to_jl()
        jl_new_route = jl.sample(jl_kernel, jl_route)
        return Route.from_jl(jl_new_route)


@dataclass
class InsertionKernel(AbstractKernel):
    """Python wrapper for Julia InsertionKernel.
    
    Kernel that removes a city and inserts it at another position.
    
    Attributes:
        seed: Random seed (optional)
    """
    seed: int | None = None

    @t.override
    def _to_jl_impl(self) -> AnyValue:
        """Convert to Julia InsertionKernel."""
        if self.seed is not None:
            return jl.InsertionKernel(self.seed)
        return jl.InsertionKernel()

    @classmethod
    @t.override
    def from_jl(cls, jl_value: AnyValue) -> t.Self:
        """Create InsertionKernel from Julia InsertionKernel."""
        seed = jl.TspJulia.Kernels.get_seed(jl_value)
        return cls(seed=int(seed) if seed is not None else None)

    @t.override
    def sample(self, route: Route) -> Route:
        """Sample a new route by inserting a city at a different position."""
        jl_kernel = self.to_jl()
        jl_route = route.to_jl()
        jl_new_route = jl.sample(jl_kernel, jl_route)
        return Route.from_jl(jl_new_route)


@dataclass
class RandomWalkKernel(AbstractKernel):
    """Python wrapper for Julia RandomWalkKernel.
    
    Kernel that randomly selects from swap, reversion, or insertion.
    
    Attributes:
        seed: Random seed (optional)
    """
    seed: int | None = None

    @t.override
    def _to_jl_impl(self) -> AnyValue:
        """Convert to Julia RandomWalkKernel."""
        if self.seed is not None:
            return jl.RandomWalkKernel(self.seed)
        return jl.RandomWalkKernel()

    @classmethod
    @t.override
    def from_jl(cls, jl_value: AnyValue) -> t.Self:
        """Create RandomWalkKernel from Julia RandomWalkKernel."""
        seed = jl.TspJulia.Kernels.get_seed(jl_value)
        return cls(seed=int(seed) if seed is not None else None)

    @t.override
    def sample(self, route: Route) -> Route:
        """Sample a new route using a random kernel."""
        jl_kernel = self.to_jl()
        jl_route = route.to_jl()
        jl_new_route = jl.sample(jl_kernel, jl_route)
        return Route.from_jl(jl_new_route)


@dataclass
class MixingKernel(AbstractKernel):
    """Python wrapper for Julia MixingKernel.
    
    Kernel that mixes multiple kernels with given probabilities.
    
    Attributes:
        kernels: List of kernel instances
        probs: List of probabilities (must sum to 1)
        seed: Random seed (optional)
    """
    kernels: t.Sequence[AbstractKernel]
    probs: list[float]
    seed: int | None = None

    @t.override
    def _to_jl_impl(self) -> AnyValue:
        """Convert to Julia MixingKernel."""
        jl_kernels = [k.to_jl() for k in self.kernels]
        jl_kernels = jl.seval("x -> Vector{AbstractKernel}(x)")(jl_kernels)
        jl_probs = jl.seval("x -> Vector{Float64}(x)")(self.probs)
        if self.seed is not None:
            return jl.MixingKernel(jl_kernels, jl_probs, self.seed)
        return jl.MixingKernel(jl_kernels, jl_probs)

    @classmethod
    @t.override
    def from_jl(cls, jl_value: AnyValue) -> t.Self:
        """Create MixingKernel from Julia MixingKernel."""
        # This is more complex as we need to identify kernel types
        # For now, we'll raise NotImplementedError
        raise NotImplementedError("from_jl not yet implemented for MixingKernel")

    @t.override
    def sample(self, route: Route) -> Route:
        """Sample a new route using one of the mixed kernels."""
        jl_kernel = self.to_jl()
        jl_route = route.to_jl()
        jl_new_route = jl.sample(jl_kernel, jl_route)
        return Route.from_jl(jl_new_route)
