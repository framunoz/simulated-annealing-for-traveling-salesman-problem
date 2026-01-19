
import pytest
from juliacall import AnyValue  # type: ignore

from tsp.jl.elements import Route
from tsp.jl.kernels import (
    InsertionKernel,
    MixingKernel,
    RandomWalkKernel,
    ReversionKernel,
    SwapKernel,
)


@pytest.mark.parametrize("kernel_cls", [
    SwapKernel,
    ReversionKernel,
    InsertionKernel,
    RandomWalkKernel
])
def test_simple_kernels_roundtrip(kernel_cls):
    # Test with seed
    kernel = kernel_cls(seed=42)
    jl_kernel = kernel.to_jl()
    
    kernel_back = kernel_cls.from_jl(jl_kernel)
    assert kernel_back.seed == 42
    assert kernel_back == kernel
    
    # Test without seed
    kernel_no_seed = kernel_cls(seed=None)
    jl_kernel_ns = kernel_no_seed.to_jl()
    
    kernel_back_ns = kernel_cls.from_jl(jl_kernel_ns)
    # Depending on implementation, it might come back as None or some state
    # But usually checks if seed property exists/is set using helper
    # Our implementation: seed=int(seed) if seed is not None else None
    assert kernel_back_ns.seed is None or isinstance(kernel_back_ns.seed, int)


def test_kernel_sampling():
    # Setup a route of length 5: 0->1->2->3->4
    route = Route([0, 1, 2, 3, 4])
    
    kernel = SwapKernel(seed=123)
    
    # Sample new route
    new_route = kernel.sample(route)
    
    assert isinstance(new_route, Route)
    assert len(new_route) == 5
    # Swap should preserve elements, just order changes
    assert sorted(new_route.route) == sorted(route.route)
    # Unlikely to be exactly same with swap and enough length
    assert new_route != route

def test_mixing_kernel_to_jl():
    kernels = [SwapKernel(), ReversionKernel()]
    probs = [0.4, 0.6]
    mixing_kernel = MixingKernel(kernels=kernels, probs=probs, seed=55)
    
    jl_mixing = mixing_kernel.to_jl()
    # We can at least validity check the conversion succeeded
    assert jl_mixing is not None
    
    # Sampling with mixed kernel
    route = Route([0, 1, 2, 3, 4])
    new_route = mixing_kernel.sample(route)
    assert isinstance(new_route, Route)
    assert len(new_route) == 5

def test_mixing_kernel_from_jl_not_implemented():
    kernels = [SwapKernel(), ReversionKernel()]
    probs = [0.5, 0.5]
    mixing_kernel = MixingKernel(kernels=kernels, probs=probs)
    jl_mixing = mixing_kernel.to_jl()
    
    with pytest.raises(NotImplementedError):
        MixingKernel.from_jl(jl_mixing)
