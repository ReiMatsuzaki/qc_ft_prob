"""
Test FSL encoding with coordinate mapping fixes.
"""
import numpy as np
from scipy.stats import norm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from newsvendor import analyze_fsl_encoding, encode_demand_distribution

def test_fsl_with_different_M():
    """Test FSL encoding with different M values."""
    print("=" * 70)
    print("Testing FSL Encoding with Coordinate Mapping Fixes")
    print("=" * 70)

    # Gaussian demand distribution
    mu, sigma = 3, 1
    D_max = 5
    demand_pdf = lambda x: norm.pdf(x, mu, sigma)

    # Also create discrete version for comparison
    demand_dist = {}
    for d in range(D_max + 1):
        demand_dist[d] = norm.pdf(d, mu, sigma)
    total = sum(demand_dist.values())
    demand_dist = {d: p/total for d, p in demand_dist.items()}

    print(f"\nTarget distribution (discrete Gaussian, μ={mu}, σ={sigma}, D_max={D_max}):")
    for d in sorted(demand_dist.keys()):
        print(f"  d={d}: {demand_dist[d]:.4f}")

    # Test with different M values
    M_values = [1, 2, 4, 8]

    for M in M_values:
        print(f"\n{'=' * 70}")
        print(f"Testing with M={M}")
        print(f"{'=' * 70}")

        try:
            # Encode distribution
            ks, cs, meta = encode_demand_distribution(demand_pdf, D_max, M=M)
            print(f"\nFourier coefficients computed:")
            print(f"  Modes (ks): {ks}")
            print(f"  Number of modes: {len(ks)}")

            # Analyze encoding
            fsl_metrics = analyze_fsl_encoding(
                ks, cs, demand_dist, D_max=D_max, n_samples=10000
            )

            print(f"\nFSL Metrics:")
            print(f"  Classical Fidelity: {fsl_metrics['fidelity']:.6f}")
            print(f"  Total Variation Distance: {fsl_metrics['tvd']:.6f}")

            print(f"\nEncoded distribution:")
            encoded_probs = fsl_metrics['encoded_probs']
            for d in range(D_max + 1):
                target_prob = demand_dist.get(d, 0.0)
                encoded_prob = encoded_probs.get(d, 0.0)
                diff = encoded_prob - target_prob
                print(f"  d={d}: target={target_prob:.4f}, encoded={encoded_prob:.4f}, diff={diff:+.4f}")

            # Check probability mass in valid range
            total_encoded = sum(encoded_probs.get(d, 0) for d in range(D_max + 1))
            print(f"\nProbability mass in [0, {D_max}]: {total_encoded:.2%}")

            if fsl_metrics['fidelity'] > 0.95:
                print("✓ GOOD: Fidelity > 0.95")
            elif fsl_metrics['fidelity'] > 0.90:
                print("⚠ ACCEPTABLE: Fidelity 0.90-0.95")
            else:
                print("✗ POOR: Fidelity < 0.90")

        except Exception as e:
            print(f"✗ FAILED with M={M}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("Test Complete")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    test_fsl_with_different_M()
