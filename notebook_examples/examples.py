# %% [markdown]
# # Fitting a simple model
# 
# This is a notebook to exemplify the use of the `tsp.sa.SimulatedAnnealingTSP` class. This class uses a Markov Chain Monte Carlo (MCMC) type algorithm called Simulated Annealing (SA) to estimate the optimal route in the Traveling Salesman Problem (TSP).
# 
# To exemplify the use of this class, we will start by generating a toy dataset.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tsp.utils import sample_coordinates
from tsp.plotting import plot_route

seed = 99301023485
rng = np.random.default_rng(seed)

coords = sample_coordinates(50, seed=rng)

plot_route(coords)
imgs_path = Path("../imgs")
imgs_path.mkdir(exist_ok=True)
plt.savefig(imgs_path / "route.png")
plt.show()

# %% [markdown]
# From these coordinates, the associated distance matrix is

# %%
from tsp.utils import distance_matrix

dist_matrix = distance_matrix(coords)
dist_matrix.shape

# %% [markdown]
# An instance of the `SimulatedAnnealingTSP` class shall now be created. This class implements the SA algorithm applied specifically to the TSP. To create an instance of this class, it is sufficient to provide the matrix of distances between the routes.

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
# And the model will be run to find the optimal route. By default, the model is limited to 1,000,000 iterations, but will stop early if no changes have been accepted after 10,000 iterations.

# %%
%%time
best_route, best_val = proxy_sa.run()

# %% [markdown]
# Through the `best_value` and `best_route` properties, the best value and associated best route of the algorithm can be found.

# %%
best_val

# %% [markdown]
# In the next cell you can see the optimal route of the algorithm plotted.

# %%
plot_route(coords, route=best_route)
plt.savefig(imgs_path / "best_route.png")
plt.show()

# %% [markdown]
# And you can also see a summary of how the algorithm performed over the iterations.

# %%
from tsp.plotting import plot_summary_sa

plot_summary_sa(proxy_sa)
plt.savefig(imgs_path / "summary_sa.png")
plt.show()

# %% [markdown]
# Notice that the route can be improved, and this is because by default, the algorithm uses a kernel that performs a swap between two cities in the route. The next section will show how to improve the performance of the algorithm by changing the kernel.

# %% [markdown]
# # Tuning the parameters

# %% [markdown]
# The performance of the algorithm can be improved by modifying some parameters of the class. Let's start by talking about the *kernels*: the kernels in the algorithm allow to decide the way in which a "jump" will be made in the graph in which the SA is performed. Or in simpler words, it allows to decide how a modification of the previous route will be made, which will be the next candidate in the SA.
# 
# The available kernels can be found in the `tsp.kernels` module, where there is also the possibility of mixing between several kernels through the `MixingKernelTSP`, where there is also the possibility to choose the weights assigned to each kernel. An example of the use of this class can be seen below:

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
# And you can use this kernel in the `SimulatedAnnealingTSP` class. Additionally, you can provide a cooling schedule function, or instantiate existing functions in `tsp.utils`, by modifying the initial parameters. Further information can be found in the `SimulatedAnnealingTSP` documentation.

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
# We run the model...

# %%
%%time
best_route_tuned, best_val_tuned = proxy_tuned.run()

# %% [markdown]
# And we look at the best value that optimisation provides:

# %%
best_val_tuned

# %% [markdown]
# It can be noted that the value of the total distance that the route realised is significantly better than in the previous case. If we look at the map, we can see that the solution looks better than the previous solution.

# %%
plot_route(coords, route=best_route_tuned)
plt.savefig(imgs_path / "best_route_tuned.png")
plt.show()

# %%
plot_summary_sa(proxy_tuned)
plt.savefig(imgs_path / "summary_tuned.png")
plt.show()

# %%



