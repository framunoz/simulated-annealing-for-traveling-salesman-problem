# %% [markdown]
# # Traveling Salesman Problem with Simulated Annealing

# Welcome to this tutorial on using the `tsp` library to solve the Traveling Salesman Problem (TSP) using Simulated Annealing (SA).

# **What is the Traveling Salesman Problem?**
# The TSP is a classic algorithmic problem in the fields of computer science and operations research. It asks: "Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?"

# **What is Simulated Annealing?**
# Simulated Annealing is a probabilistic technique for approximating the global optimum of a given function. It is inspired by the process of annealing in metallurgy, where a material is heated and then slowly cooled to decrease defects and minimize the system's energy.

# In our context:
# - **Energy** corresponds to the total distance of the route (we want to minimize this).
# - **Temperature** controls the probability of accepting a worse solution. High temperatures allow the algorithm to explore the solution space broadly (avoiding local minima), while low temperatures focus on exploitation (refining the best solution found).

# This notebook will guide you through generating data, fitting a model, and visualizing the results.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tsp.utils import sample_coordinates
from tsp.plotting import plot_route

seed = 99301023485
rng = np.random.default_rng(seed)

# We start by generating a synthetic dataset of 50 cities with random coordinates.
coords = sample_coordinates(50, seed=rng)

# Visualize the initial random arrangement of cities
plot_route(coords)
imgs_path = Path("../imgs")
imgs_path.mkdir(exist_ok=True)
plt.savefig(imgs_path / "route.png")
plt.show()

# %% [markdown]
# ## 1. Pre-calculating Distances

# To optimize the performance of the SA algorithm, it is crucial to pre-calculate the distances between all pairs of cities. This creates a distance matrix which allows for O(1) lookups during the optimization process, rather than re-calculating Euclidean distances at every step.

# %%
from tsp.utils import distance_matrix

dist_matrix = distance_matrix(coords)
dist_matrix.shape

# %% [markdown]
# ## 2. Initializing the Model

# Now we initialize the `SimulatedAnnealingTSP` class. This is the core orchestrator of the optimization.

# Key components:
# - **`Cities`**: A container object that holds the coordinate points (`Point`).
# - **`kernel`**: The strategy used to propose a new candidate solution (neighbor) from the current solution.
#     - Here we use `SwapKernel`: A simple strategy that randomly swaps two cities in the route order.

# This standard setup provides a baseline for optimization.

# %%
import tsp.py as tsp

points = [tsp.Point(c[0], c[1]) for c in coords]
cities = tsp.Cities(points)

model_sa = tsp.SimulatedAnnealingTSP(
    cities,
    kernel=tsp.SwapKernel(seed=seed),
)
proxy_sa = tsp.SAStatsProxy(model_sa)

# %% [markdown]
# ## 3. Running the Optimization

# We execute the `run()` method to start the Simulated Annealing process.

# **Stopping Criteria:**
# The model is configured with defaults (which can be overridden):
# - **Max Iterations**: 1,000,000 (hard limit)
# - **Early Stopping**: Stops if no improvement is accepted after a specified number of consecutive iterations (default 10,000), indicating convergence.

# %%
%%time
best_route, best_val = proxy_sa.run()

# %% [markdown]
# ## 4. Analyzing Results

# The `proxy_sa.run()` method returns:
# - **`best_route`**: The sequence of city indices that minimizes the total distance.
# - **`best_val`**: The total distance of that optimal route.

# Let's inspect the minimum total distance found:

# %%
best_val

# %% [markdown]
# ## 5. Visualization

# Visualizing the solution is critical for validating the result. A "good" TSP solution typically avoids crossing lines (edges) and looks like a coherent loop.

# %%
plot_route(coords, route=best_route)
plt.savefig(imgs_path / "best_route.png")
plt.show()

# %% [markdown]
# ## 6. Convergence Analysis

# We can examine how the solution improved over time. The summary plot typically shows:
# - **Trace**: The "Energy" (distance) at each step.
# - **Temperature**: How the acceptance probability decreased over time.

# A healthy trace shows a rapid decrease in distance initially, followed by a plateau as the system "cools" and settles into a minimum.

# %%
from tsp.plotting import plot_summary_sa

plot_summary_sa(proxy_sa)
plt.savefig(imgs_path / "summary_sa.png")
plt.show()

# %% [markdown]
# ---
# # Advanced: Hyperparameter Tuning

# While the basic `SwapKernel` works, we can often achieve better results or faster convergence by "tuning" the SA parameters.

# ## Kernels (Neighborhood Functions)
# The choice of kernel determines how we explore the solution space (the "landscape").
# - **`SwapKernel`**: Swaps two random cities. Good for small adjustments.
# - **`ReversionKernel`**: Reverses a subsequence of the route. Very effective for untangling "knots" or crossing lines in 2D Euclidean TSP.
# - **`InsertionKernel`**: Moves a city to a new position.

# ## Mixing Kernels
# We can combine these strategies using `MixingKernel`. By assigning probabilities (weights), we can create a robust search strategy that benefits from the strengths of each kernel.

# Below, we define a mixture with:
# - **50% Reversion**: Aggressive untangling.
# - **30% Insertion**: Structural changes.
# - **20% Swap**: Minor tweaks.
# - **0.05% RandomWalk**: Pure exploration (optional).

# %%
mixing_kernel = tsp.MixingKernel(
    kernels=[
        (20, tsp.SwapKernel(seed=seed)),
        (50, tsp.ReversionKernel(seed=seed)),
        (30, tsp.InsertionKernel(seed=seed)),
        (0.05, tsp.RandomWalkKernel(seed=seed)),
    ],
    seed=seed,
)

# %% [markdown]
# ## Cooling Schedule

# The `temperature` parameter defines the cooling schedule. A slower cooling rate (e.g., higher `rho` in exponential cooling) allows more exploration but requires more iterations to converge.

# - **`exponential_cooling_schedule(rho=0.9999)`**: Decays the temperature very slowly ($T_{k+1} = T_k \times \rho$). This gives the algorithm ample time to escape local minima.

# We construct a new model with these advanced settings.

# %%
from tsp.utils import exponential_cooling_schedule

model_sa_tuned = tsp.SimulatedAnnealingTSP(
    cities=cities,
    n_iter=100_000,
    kernel=mixing_kernel,
    temperature=exponential_cooling_schedule(rho=0.9999),
    early_stop=True,
    stop_after=10_000,
)

proxy_tuned = tsp.SAStatsProxy(model_sa_tuned)

# %% [markdown]
# Running the optimized model...

# %%
%%time
best_route_tuned, best_val_tuned = proxy_tuned.run()

# %% [markdown]
# ### Comparison
# Let's see the new best distance. It should be lower (better) than our initial run.

# %%
best_val_tuned

# %% [markdown]
# Visualizing the improved solution:

# %%
plot_route(coords, route=best_route_tuned)
plt.savefig(imgs_path / "best_route_tuned.png")
plt.show()

# %%
plot_summary_sa(proxy_tuned)
plt.savefig(imgs_path / "summary_tuned.png")
plt.show()

# %%
