
import abc
import typing as t
from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator

from ..plotting import SAStatsProtocol
from ..utils import _EPS, cooling_schedule_function, exponential_cooling_schedule
from .elements import Cities, Route
from .kernels import Kernel, SwapKernel

__all__ = ["SimulatedAnnealingTSP", "SAStatsProxy"]


@dataclass
class AbstractSimulatedAnnealing(abc.ABC):
    """
    Abstract base class for Simulated Annealing.
    """
    
    @abc.abstractmethod
    def initial_route(self) -> tuple[Route, float]:
        ...
        
    @abc.abstractmethod
    def sample(self, route: Route) -> tuple[Route, float]:
        ...
        
    @abc.abstractmethod
    def acceptance_probability(self, current_val: float, new_val: float, k: int) -> float:
        ...
        
    @abc.abstractmethod
    def stop_criteria(self, stop_counter: int) -> bool:
        ...
        
    @property
    @abc.abstractmethod
    def n_iter(self) -> int:
        ...
        
    @property
    @abc.abstractmethod
    def rng(self) -> Generator:
        ...
        
    def on_step(self, k: int, route: Route, val: float, log_accept_prob: float):
        pass
        
    def on_accept(self, k: int, route: Route, val: float):
        pass

    def run(self) -> tuple[Route, float]:
        route, route_val = self.initial_route()
        
        best_route = route
        best_val = route_val
        
        stop_counter = 0
        current_route = route
        current_val = route_val
        
        for k in range(1, self.n_iter + 1):
            new_route, new_val = self.sample(current_route)
            
            log_accept_prob = self.acceptance_probability(current_val, new_val, k)
            
            if log_accept_prob >= 0 or self.rng.uniform() < np.exp(log_accept_prob):
                # Accepted
                stop_counter = 0
                
                if new_val < best_val:
                    best_route = new_route
                    best_val = new_val
                    
                current_route = new_route
                current_val = new_val
                
                self.on_accept(k, current_route, current_val)
            else:
                stop_counter += 1
                
            self.on_step(k, current_route, current_val, log_accept_prob)
            
            if self.stop_criteria(stop_counter):
                break
                
        return best_route, best_val


@dataclass
class SimulatedAnnealingTSP(AbstractSimulatedAnnealing):
    """
    The Simulated Annealing algorithm for solving the Traveling Salesman Problem.
    Pure Python implementation.
    """
    cities: Cities
    kernel: Kernel
    n_iter: int = 1_000_000
    temperature: cooling_schedule_function = field(default_factory=exponential_cooling_schedule)
    early_stop: bool = True
    stop_after: int = 10_000
    seed: t.Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
             self._rng = np.random.default_rng(self.seed)
        else:
             if hasattr(self.kernel, "rng"):
                 self._rng = self.kernel.rng
             else:
                 self._rng = np.random.default_rng()

    @property
    def rng(self) -> Generator:
        return self._rng

    def initial_route(self) -> tuple[Route, float]:
        route = Route(list(range(len(self.cities))))
        return route, self.cities.total_distance(route)

    def sample(self, route: Route) -> tuple[Route, float]:
        new_route = self.kernel.sample(route)
        return new_route, self.cities.total_distance(new_route)

    def acceptance_probability(self, current_val: float, new_val: float, k: int) -> float:
        gap = new_val - current_val
        if gap <= 0:
            return 0.0
        return -gap / (self.temperature(k) + _EPS)

    def stop_criteria(self, stop_counter: int) -> bool:
        return self.early_stop and stop_counter > self.stop_after


@dataclass
# SAStatsProxy follows SAStatsProtocol implicitly via fields
class SAStatsProxy(AbstractSimulatedAnnealing):
    """
    Proxy wrapper that collects statistics during SA execution.
    """
    sa: SimulatedAnnealingTSP
    
    # Public fields for statistics
    log_acceptance_probs: list[float] = field(default_factory=list, init=False)
    n_accept: int = field(default=0, init=False)
    acceptance_ratio: list[float] = field(default_factory=list, init=False)
    values: list[float] = field(default_factory=list, init=False)
    # accept_prob protocol requirement: Sequence[float]. 
    # We can compute it on the fly or store it. Keeping it as property might be cleaner if user allows,
    # but request said "campos directos". I'll add a field.
    accept_prob: list[float] = field(default_factory=list, init=False)

    def __post_init__(self):
        # Clear/Init stats
        self.log_acceptance_probs = []
        self.n_accept = 0
        self.acceptance_ratio = []
        self.values = []
        self.accept_prob = []

    # Delegates
    @property
    def n_iter(self) -> int:
        return self.sa.n_iter
        
    @property
    def rng(self) -> Generator:
        return self.sa.rng
        
    def initial_route(self) -> tuple[Route, float]:
        return self.sa.initial_route()
        
    def sample(self, route: Route) -> tuple[Route, float]:
        return self.sa.sample(route)
        
    def acceptance_probability(self, current_val: float, new_val: float, k: int) -> float:
        return self.sa.acceptance_probability(current_val, new_val, k)
        
    def stop_criteria(self, stop_counter: int) -> bool:
        return self.sa.stop_criteria(stop_counter)

    # Hooks for stats
    def on_accept(self, k: int, route: Route, val: float):
        self.n_accept += 1

    def on_step(self, k: int, route: Route, val: float, log_accept_prob: float):
        self.log_acceptance_probs.append(log_accept_prob)
        self.values.append(val)
        self.acceptance_ratio.append(self.n_accept / k)
        self.accept_prob.append(np.exp(log_accept_prob))
        
    def run(self) -> tuple[Route, float]:
        # Reset stats before run
        self.__post_init__()
        return super().run()
