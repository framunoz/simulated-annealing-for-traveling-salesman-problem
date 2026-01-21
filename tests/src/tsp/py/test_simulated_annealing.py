
import numpy as np
import pytest

from tsp.py.elements import Cities, Point, Route
from tsp.py.kernels import SwapKernel
from tsp.py.simulated_annealing import SAStatsProxy, SimulatedAnnealingTSP


@pytest.fixture
def simple_problem():
    # Square 1x1
    # 0: (0,0), 1: (1,0), 2: (1,1), 3: (0,1)
    # 0-1: 1, 1-2: 1, 2-3: 1, 3-0: 1
    # Perimeter: 4
    
    points = [
        Point(0.0, 0.0),
        Point(1.0, 0.0),
        Point(1.0, 1.0),
        Point(0.0, 1.0)
    ]
    
    cities = Cities(points)
    kernel = SwapKernel(seed=42)
    return cities, kernel

def test_sa_run_basic(simple_problem):
    cities, kernel = simple_problem
    sa = SimulatedAnnealingTSP(
        cities=cities,
        kernel=kernel,
        n_iter=1000,
        early_stop=True,
        stop_after=100,
        seed=123
    )
    
    best_route, best_val = sa.run()
    
    assert isinstance(best_route, Route)
    assert len(best_route) == 4
    assert isinstance(best_val, float)
    assert best_val > 0.0
    # Optimal for unit square is 4.0 (perimeter)
    assert best_val >= 4.0

def test_sa_stats_proxy(simple_problem):
    cities, kernel = simple_problem
    sa = SimulatedAnnealingTSP(
        cities=cities,
        kernel=kernel,
        n_iter=500,
        seed=123
    )
    
    proxy = SAStatsProxy(sa)
    
    # Run via proxy
    best_route, best_val = proxy.run()
    
    # Check stats are populated
    vals = proxy.values
    assert isinstance(vals, list)
    assert len(vals) > 0
    
    n_accept = proxy.n_accept
    assert n_accept >= 0
    
    ratios = proxy.acceptance_ratio
    assert len(ratios) == len(vals)
    
    log_probs = proxy.log_acceptance_probs
    assert len(log_probs) == len(vals)
    
    accept_prob = proxy.accept_prob
    assert len(accept_prob) == len(vals)
    assert all(0.0 <= p <= 1.0 + 1e-9 for p in accept_prob)
    
    assert all(0.0 <= p <= 1.0 + 1e-9 for p in accept_prob)
    
    # assert proxy.best_value == best_val # Removed property

def test_sa_initialization_defaults(simple_problem):
    cities, kernel = simple_problem
    sa = SimulatedAnnealingTSP(cities, kernel)
    assert sa.n_iter == 1_000_000
    assert sa.early_stop is True
