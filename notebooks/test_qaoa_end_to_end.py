"""
End-to-end test of QAOA newsvendor solver with FSL encoding fixes.
"""
import numpy as np
from scipy.stats import norm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from newsvendor import solve_newsvendor_qaoa, comprehensive_analysis

def test_small_problem():
    """Test with a small problem to verify end-to-end functionality."""
    print("=" * 70)
    print("End-to-End QAOA Newsvendor Test (Small Problem)")
    print("=" * 70)

    # Gaussian demand distribution
    mu, sigma = 3, 1
    D_max = 5
    Q_max = 5
    demand_pdf = lambda x: norm.pdf(x, mu, sigma)

    print(f"\nProblem setup:")
    print(f"  Demand: Gaussian(μ={mu}, σ={sigma})")
    print(f"  Q_max={Q_max}, D_max={D_max}")
    print(f"  Order cost (c): 1.0")
    print(f"  Stockout penalty (λ): 5.0")
    print(f"  QAOA depth (p): 1")
    print(f"  Fourier truncation (M): 4")

    try:
        # Run QAOA solver
        result = solve_newsvendor_qaoa(
            demand_dist=demand_pdf,
            c=1.0,
            lam=5.0,
            Q_max=Q_max,
            D_max=D_max,
            p=1,
            M=4,
            n_shots=500,  # Fewer shots for speed
            verbose=True
        )

        print("\n" + "=" * 70)
        print("RESULT SUMMARY")
        print("=" * 70)
        print(f"Quantum solution: q = {result['quantum_solution']}")
        print(f"Quantum cost: {result['quantum_cost']:.4f}")
        print(f"Classical solution: q = {result['classical_solution']}")
        print(f"Classical cost: {result['classical_cost']:.4f}")
        print(f"Approximation ratio: {result['quantum_cost'] / result['classical_cost']:.4f}")
        print(f"Measurement confidence: {result.get('confidence', 'N/A')}")

        # Check if the solution is reasonable
        if result['quantum_solution'] in range(Q_max + 1):
            print("\n✓ Quantum solution is within valid range")
        else:
            print(f"\n✗ WARNING: Quantum solution {result['quantum_solution']} is outside [0, {Q_max}]")

        if result['quantum_cost'] / result['classical_cost'] <= 1.1:
            print("✓ Quantum cost is close to classical optimal (within 10%)")
        else:
            print(f"⚠ Quantum cost is {result['quantum_cost'] / result['classical_cost']:.1%} of classical optimal")

        # Run comprehensive analysis
        print("\n" + "=" * 70)
        print("COMPREHENSIVE ANALYSIS")
        print("=" * 70)
        comprehensive_analysis(result)

        print("\n" + "=" * 70)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nGenerated visualization files:")
        print("  - qaoa_convergence.png")
        print("  - fsl_fidelity.png")
        print("  - newsvendor_results.png")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_small_problem()
