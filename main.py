from juliacall import Main as jl

jl.include("src/TspJulia/elements/Elements.jl")
jl.seval("using .Elements")
jl.include("src/TspJulia/kernels/Kernels.jl")
jl.seval("using .Kernels")

p = jl.Point(1.0, 2.0)
print(p)
print(type(p))

c = jl.Cities([jl.Point(1.0, 2.0), jl.Point(3.0, 4.0)])
print(c)
print(type(c))

dm = jl.compute_distance_matrix(c)
print(dm)
print(type(dm))

print(dm[0, 1])
print(type(dm[0, 1]))

r = jl.Route([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(r)
print(type(r))

k1 = jl.SwapKernelTSP()
print(k1)
print(type(k1))

r1 = jl.sample(k1, r)
print(r1)
print(type(r1))

k2 = jl.ReversionKernelTSP()
print(k2)
print(type(k2))

r2 = jl.sample(k2, r)
print(r2)
print(type(r2))

k3 = jl.InsertionKernelTSP()
print(k3)
print(type(k3))

r3 = jl.sample(k3, r)
print(r3)
print(type(r3))

k4 = jl.RandomWalkKernelTSP()
print(k4)
print(type(k4))

r4 = jl.sample(k4, r)
print(r4)
print(type(r4))

k5 = jl.MixingKernelTSP([k1, k2, k3, k4], [0.1, 0.2, 0.3, 0.4])
print(k5)
print(type(k5))

r5 = jl.sample(k5, r)
print(r5)
print(type(r5))
