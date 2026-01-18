
# Evaluate the module file in the global scope of the current module (Main)
include("src/TspJulia/elements/Elements.jl")
include("src/TspJulia/kernels/Kernels.jl")
include("src/TspJulia/ScheduleFns.jl")
include("src/TspJulia/SimulatedAnnealing.jl")
# Bring exported names into scope. The dot is for a locally defined module.
using .Elements
using .Kernels
using .ScheduleFns
using .SimulatedAnnealing

using Random

p = Point(1.0, 2.0)
c = Cities([
    p,
    Point(2.0, 3.0),
    Point(3.0, 4.0),
    Point(4.0, 5.0),
    Point(5.0, 6.0),
    Point(6.0, 7.0),
    Point(7.0, 8.0),
    Point(8.0, 9.0),
    Point(9.0, 10.0),
    Point(10.0, 11.0),
])

r = Route(1:10)
dm = compute_distance_matrix(c)

k1 = SwapKernel(42)

r1 = sample(k1, r)

k2 = ReversionKernel()

r2 = sample(k2, r)

k3 = InsertionKernel()

r3 = sample(k3, r)

k4 = RandomWalkKernel()

r4 = sample(k4, r)

k5 = MixingKernel([k1, k2, k4], [0.1, 0.2, 0.7])

r5 = sample(k5, r)

k6 = MixingKernel([k5, k4], [0.9, 0.1], 42)

r6 = sample(k6, r)

sa = SimulatedAnnealingTSP(c, k6)
sa_stats = SAStatsProxy(sa)
println(sa)

last_route = run_sa(sa_stats)
println(last_route)
println(total_distance(c, last_route))

println(sa_stats)
