# Evaluate the module file in the global scope of the current module (Main)
include("src/TspJulia/Elements.jl")
include("src/TspJulia/Kernels.jl")
include("src/TspJulia/ScheduleFns.jl")
include("src/TspJulia/SimulatedAnnealing.jl")
# Bring exported names into scope. The dot is for a locally defined module.
using .Elements
using .Kernels
using .ScheduleFns
using .SimulatedAnnealing

using Random

concrete_schedule = log_cooling_schedule(100, 1.)
println(concrete_schedule)
println(concrete_schedule(1))
println(concrete_schedule(2))
println(concrete_schedule(3))
println(concrete_schedule(4))
println(concrete_schedule(5))


p = Point(1.0, 2.0)
println(p)


c = Cities([p, Point(2.0, 3.0), Point(3.0, 4.0), Point(4.0, 5.0), Point(5.0, 6.0), Point(6.0, 7.0), Point(7.0, 8.0), Point(8.0, 9.0), Point(9.0, 10.0), Point(10.0, 11.0)])
println(c)


r = Route(1:10)
println(r)


dm = compute_distance_matrix(c)
println(dm)


k1 = SwapKernelTSP(42)
println(k1)

println(_sample_swap(r, 3, 7))


r1 = sample(k1, r)
println(r1)


k2 = ReversionKernelTSP()
println(k2)

println(_sample_reversion(r, 3, 7))

r2 = sample(k2, r)
println(r2)


k3 = InsertionKernelTSP()
println(k3)

println(_sample_insertion(r, 3, 7))

r3 = sample(k3, r)
println(r3)


k4 = RandomWalkKernelTSP()
println(k4)


r4 = sample(k4, r)
println(r4)


k5 = MixingKernelTSP([k1, k2, k4], [0.1, 0.2, 0.7])
println(k5)


r5 = sample(k5, r)
println(r5)

k6 = MixingKernelTSP([k5, k4], [0.9, 0.1],)
println(k6)

r6 = sample(k6, r)
println(r6)


sa = SimulatedAnnealingTSP(c, k6)
println(sa)

println(run_sa(sa))
