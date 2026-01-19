"""Test script to verify Julia wrappers."""

from tsp.jl import (
    Cities,
    Point,
    SAStatsProxy,
    SimulatedAnnealingTSP,
    SwapKernel,
)

# Create some test cities
cities = Cities(
    cities=[
        Point(0.0, 0.0),
        Point(1.0, 1.0),
        Point(2.0, 2.0),
        Point(3.0, 0.0),
        Point(4.0, 1.0),
    ]
)

print(f"Created {len(cities)} cities")
print(f"First city: {cities[0]}")

# Create a kernel
kernel = SwapKernel(seed=42)
print(f"Created kernel: {kernel}")

# Create SA instance
sa = SimulatedAnnealingTSP(
    cities=cities,
    kernel=kernel,
    n_iter=1000,
    early_stop=True,
    stop_after=100,
)
print("Created SA instance")

# Wrap with stats proxy
sa_stats = SAStatsProxy(sa=sa)
print("Created stats proxy")

# Run the algorithm
print("Running SA...")
best_route, best_value = sa_stats.run()

print("\nResults:")
print(f"Best route: {best_route.route}")
print(f"Best value: {best_value:.4f}")
print(f"Acceptance ratio: {sa_stats.acceptance_ratio[-1]:.2%}")
print(f"Total accepted: {sa_stats.n_accept}")
print(f"Total iterations: {len(sa_stats.values)}")

print("\n✓ All tests passed!")
