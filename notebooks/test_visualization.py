"""
Test script for visualization functions.
"""
import numpy as np
from scipy.stats import norm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from newsvendor import (
    solve_newsvendor_qaoa,
    comprehensive_analysis,
    plot_optimization_convergence,
    visualize_circuit_structure,
    analyze_fsl_encoding
)

def test_visualization():
    """Test all visualization functions with a small problem."""
    print("=" * 70)
    print("Testing Visualization Functions")
    print("=" * 70)

    # Simple Gaussian demand
    mu, sigma = 3, 1
    demand_pdf = lambda x: norm.pdf(x, mu, sigma)

    print("\n1. Running QAOA...")
    result = solve_newsvendor_qaoa(
        demand_dist=demand_pdf,
        c=1.0, lam=5.0,
        Q_max=5, D_max=5,
        p=1, M=1,
        n_shots=500,  # Fewer shots for speed
        verbose=True
    )

    print("\n2. Testing individual visualization functions...")

    try:
        print("\n  a) Testing optimization convergence plot...")
        plot_optimization_convergence(result['optimization_history'])
        print("     ✓ Convergence plot succeeded")
    except Exception as e:
        print(f"     ✗ Convergence plot failed: {e}")

    try:
        print("\n  b) Testing circuit structure visualization...")
        visualize_circuit_structure(result, show_full=False)
        print("     ✓ Circuit structure visualization succeeded")
    except Exception as e:
        print(f"     ✗ Circuit structure visualization failed: {e}")

    try:
        print("\n  c) Testing FSL encoding analysis...")
        fsl_metrics = analyze_fsl_encoding(
            result['ks'], result['cs'],
            result['demand_dist'],
            D_max=5,
            n_samples=1000  # Fewer samples for speed
        )
        print(f"     ✓ FSL analysis succeeded (fidelity: {fsl_metrics['fidelity']:.4f})")
    except Exception as e:
        print(f"     ✗ FSL analysis failed: {e}")

    print("\n3. Testing comprehensive analysis...")
    try:
        comprehensive_analysis(result)
        print("     ✓ Comprehensive analysis succeeded")
    except Exception as e:
        print(f"     ✗ Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Visualization Test Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - qaoa_convergence.png")
    print("  - fsl_fidelity.png")
    print("  - newsvendor_results.png")
    print("  - (circuit_diagram.png if qulacsvis is installed)")


if __name__ == "__main__":
    test_visualization()
