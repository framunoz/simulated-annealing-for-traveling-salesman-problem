"""Python wrappers for Julia SimulatedAnnealing module."""

import typing as t
from dataclasses import dataclass

from juliacall import AnyValue  # type: ignore
from juliacall import Main as jl  # type: ignore

from .common import JuliaWrapper
from .elements import Cities, Route
from .kernels import (
    InsertionKernel,
    MixingKernel,
    RandomWalkKernel,
    ReversionKernel,
    SwapKernel,
)

__all__ = ["SimulatedAnnealingTSP", "SAStatsProxy"]

# Initialize Julia modules
if not jl.seval("isdefined(Main, :TspJulia)"):
    jl.include("srcjl/TspJulia.jl")
jl.seval("using .TspJulia")
jl.seval("using .TspJulia.ScheduleFns")
jl.seval("using .TspJulia.SimulatedAnnealing")


KernelType = SwapKernel | ReversionKernel | InsertionKernel | RandomWalkKernel | MixingKernel


@dataclass
class SimulatedAnnealingTSP(JuliaWrapper):
    """Python wrapper for Julia SimulatedAnnealingTSP.
    
    Main structure for the Simulated Annealing algorithm applied to TSP.
    
    Attributes:
        cities: The set of cities to visit
        kernel: The kernel used to propose new routes
        n_iter: Maximum number of iterations
        temperature: Function that returns temperature at iteration k
        early_stop: Whether to stop early if no improvement is found
        stop_after: Number of iterations without improvement to trigger early stop
    """
    cities: Cities
    kernel: KernelType
    n_iter: int = 1_000_000
    temperature: t.Callable[[int], float] | None = None
    early_stop: bool = True
    stop_after: int = 10_000

    @t.override
    def _to_jl_impl(self) -> AnyValue:
        """Convert to Julia SimulatedAnnealingTSP."""
        jl_cities = self.cities.to_jl()
        jl_kernel = self.kernel.to_jl()
        
        kwargs: dict[str, t.Any] = {
            "n_iter": self.n_iter,
            "early_stop": self.early_stop,
            "stop_after": self.stop_after,
        }
        
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        
        return jl.SimulatedAnnealingTSP(jl_cities, jl_kernel, **kwargs)

    @classmethod
    @t.override
    def from_jl(cls, jl_value: AnyValue) -> t.Self:
        """Create SimulatedAnnealingTSP from Julia SimulatedAnnealingTSP."""
        # Kernel conversion is complex, skip for now
        raise NotImplementedError("from_jl not yet fully implemented for SimulatedAnnealingTSP")

    def run(self) -> tuple[Route, float]:
        """Run the simulated annealing algorithm.
        
        Returns:
            Tuple of (best_route, best_value)
        """
        jl_sa = self.to_jl()
        jl_best_route, best_value = jl.run_sa(jl_sa)
        best_route = Route.from_jl(jl_best_route)
        return best_route, float(best_value)


@dataclass
class SAStatsProxy(JuliaWrapper):
    """Python wrapper for Julia SAStatsProxy.
    
    A proxy wrapper that collects statistics during SA execution.
    
    Attributes:
        sa: The underlying SimulatedAnnealingTSP instance
        log_acceptance_probs: Logarithm of acceptance probabilities (populated after run)
        n_accept: Total number of accepted moves (populated after run)
        acceptance_ratio: Acceptance ratio at each iteration (populated after run)
        values: Objective function values at each step (populated after run)
    """
    sa: SimulatedAnnealingTSP

    @property
    def log_acceptance_probs(self) -> list[float]:
        jl_stats = self.to_jl()
        return [float(x) for x in jl_stats.log_acceptance_probs]
    
    @property
    def n_accept(self) -> int:
        jl_stats = self.to_jl()
        return int(jl_stats.n_accept)
    
    @property
    def acceptance_ratio(self) -> list[float]:
        jl_stats = self.to_jl()
        return [float(x) for x in jl_stats.acceptance_ratio]
    
    @property
    def values(self) -> list[float]:
        jl_stats = self.to_jl()
        return [float(x) for x in jl_stats.values]
    
    @t.override
    def _to_jl_impl(self) -> AnyValue:
        """Convert to Julia SAStatsProxy."""
        jl_sa = self.sa.to_jl()
        return jl.SAStatsProxy(jl_sa)

    @classmethod
    @t.override
    def from_jl(cls, jl_value: AnyValue) -> t.Self:
        """Create SAStatsProxy from Julia SAStatsProxy."""
        raise NotImplementedError("from_jl not yet implemented for SAStatsProxy")

    def run(self) -> tuple[Route, float]:
        """Run the simulated annealing algorithm with statistics collection.
        
        Returns:
            Tuple of (best_route, best_value)
        """
        jl_stats = self.to_jl()
        jl_best_route, best_value = jl.run_sa(jl_stats)
        
        best_route = Route.from_jl(jl_best_route)
        return best_route, float(best_value)
