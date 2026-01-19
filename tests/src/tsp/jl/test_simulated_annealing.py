
import pytest
from juliacall import AnyValue  # type: ignore

from tsp.jl.elements import Cities, Point, Route
from tsp.jl.kernels import SwapKernel
from tsp.jl.simulated_annealing import SAStatsProxy, SimulatedAnnealingTSP


@pytest.fixture
def simple_problem():
    # Square 1x1
    points = [
        Point(0.0, 0.0),
        Point(1.0, 0.0),
        Point(1.0, 1.0),
        Point(0.0, 1.0)
    ]
    cities = Cities(points)
    kernel = SwapKernel(seed=42)
    return cities, kernel

def test_sa_roundtrip(simple_problem):
    cities, kernel = simple_problem
    sa = SimulatedAnnealingTSP(
        cities=cities,
        kernel=kernel,
        n_iter=100,
        early_stop=False
    )
    
    jl_sa = sa.to_jl()
    assert jl_sa is not None
    
    # from_jl is not implemented yet
    with pytest.raises(NotImplementedError):
        SimulatedAnnealingTSP.from_jl(jl_sa)

def test_sa_run(simple_problem):
    cities, kernel = simple_problem
    sa = SimulatedAnnealingTSP(
        cities=cities,
        kernel=kernel,
        n_iter=1000,
        early_stop=True,
        stop_after=100
    )
    
    best_route, best_val = sa.run()
    
    assert isinstance(best_route, Route)
    assert len(best_route) == 4
    assert isinstance(best_val, float)
    assert best_val > 0.0

def test_sa_stats_proxy(simple_problem):
    cities, kernel = simple_problem
    sa = SimulatedAnnealingTSP(
        cities=cities,
        kernel=kernel,
        n_iter=500
    )
    
    proxy = SAStatsProxy(sa)
    
    # Roundtrip check (basic)
    jl_proxy = proxy.to_jl()
    assert jl_proxy is not None
    
    # Run via proxy
    best_route, best_val = proxy.run()
    
    # Check stats are populated
    # Note: These properties access the underlying Julia object fields 
    # which should be populated after run_sa!
    
    vals = proxy.values
    assert isinstance(vals, list)
    assert len(vals) > 0
    
    n_accept = proxy.n_accept
    assert n_accept >= 0
    
    ratios = proxy.acceptance_ratio
    assert len(ratios) == len(vals) - 1
    
    log_probs = proxy.log_acceptance_probs
    assert len(log_probs) == len(vals) - 1
