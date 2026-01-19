"""Test script to verify Julia object caching."""

from tsp.jl import Point


def test_caching():
    p = Point(1.0, 2.0)
    
    print("First call to to_jl()...")
    jl_obj1 = p.to_jl()
    
    print("Second call to to_jl()...")
    jl_obj2 = p.to_jl()
    
    # Check if they are the exact same object in memory (Python side reference to Julia object)
    if jl_obj1 is jl_obj2:
        print("✓ SUCCESS: Julia object is cached and the same instance is returned.")
    else:
        print("✗ FAILURE: Julia object is not the same instance.")

if __name__ == "__main__":
    test_caching()
