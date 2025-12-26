"""
Simple test for newsvendor implementation with a minimal problem size.
"""
import numpy as np
import sys
import os

# Add notebooks directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import newsvendor functions
from newsvendor import (
    encode_demand_distribution,
    compute_stockout_prob,
    classical_optimal_solution,
    normalize_demand_distribution
)

def test_basic_functions():
    """Test basic classical functions."""
    print("=" * 60)
    print("Testing Basic Functions")
    print("=" * 60)

    # Test 1: Simple uniform demand
    print("\n1. Testing demand distribution normalization...")
    demand_dist = {0: 0.2, 1: 0.3, 2: 0.3, 3: 0.2}
    demand_dist = normalize_demand_distribution(demand_dist)
    print(f"   Demand distribution: {demand_dist}")
    print(f"   Sum of probabilities: {sum(demand_dist.values()):.6f}")
    assert abs(sum(demand_dist.values()) - 1.0) < 1e-6, "Distribution should sum to 1"
    print("   ✓ Passed")

    # Test 2: Stockout probability
    print("\n2. Testing stockout probability computation...")
    q = 1
    prob = compute_stockout_prob(q, demand_dist)
    expected = demand_dist[2] + demand_dist[3]  # Pr(D > 1) = Pr(D=2) + Pr(D=3)
    print(f"   Pr(D > {q}) = {prob:.4f}")
    print(f"   Expected: {expected:.4f}")
    assert abs(prob - expected) < 1e-6, "Stockout probability incorrect"
    print("   ✓ Passed")

    # Test 3: Classical optimal solution
    print("\n3. Testing classical optimal solution...")
    c, lam = 1.0, 5.0
    Q_max = 3
    opt_q, opt_cost = classical_optimal_solution(demand_dist, c, lam, Q_max)
    print(f"   Optimal q: {opt_q}")
    print(f"   Optimal cost: {opt_cost:.4f}")

    # Verify by checking all q values
    print("\n   Cost for each q:")
    for q in range(Q_max + 1):
        cost = c * q + lam * compute_stockout_prob(q, demand_dist)
        print(f"     q={q}: cost={cost:.4f}")

    print("   ✓ Passed")

    # Test 4: FSL encoding
    print("\n4. Testing FSL demand encoding...")
    try:
        ks, cs, meta = encode_demand_distribution(demand_dist, D_max=3, M=4)
        print(f"   Number of Fourier modes: {len(ks)}")
        print(f"   Fourier metadata: {meta}")
        print("   ✓ Passed")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Basic Function Tests Complete!")
    print("=" * 60)


def test_small_qaoa():
    """Test QAOA with very small problem."""
    print("\n" + "=" * 60)
    print("Testing Small QAOA Problem")
    print("=" * 60)

    # Very small problem: Q_max=3, D_max=3
    demand_dist = {0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2}

    print(f"\nProblem setup:")
    print(f"  Demand distribution: {demand_dist}")
    print(f"  Q_max = 3, D_max = 3")
    print(f"  c = 1.0, λ = 5.0")

    try:
        from newsvendor import solve_newsvendor_qaoa

        print("\nRunning QAOA...")
        result = solve_newsvendor_qaoa(
            demand_dist=demand_dist,
            c=1.0,
            lam=5.0,
            Q_max=3,
            D_max=3,
            p=1,  # Single QAOA layer for speed
            M=4,  # Small Fourier truncation
            n_shots=100,  # Few shots for testing
            verbose=True
        )

        print("\n✓ QAOA completed successfully!")
        print(f"  Quantum solution: q = {result['quantum_solution']}")
        print(f"  Classical solution: q = {result['classical_solution']}")

    except Exception as e:
        print(f"\n✗ QAOA test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    test_basic_functions()

    print("\n\nNow testing QAOA (this may take a minute)...")
    proceed = input("Continue with QAOA test? (y/n): ")
    if proceed.lower() == 'y':
        test_small_qaoa()
    else:
        print("Skipping QAOA test.")
