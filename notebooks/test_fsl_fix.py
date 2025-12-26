"""
Test FSL encoding with IQFT fix.
"""
import numpy as np
from scipy.stats import norm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from newsvendor import analyze_fsl_encoding, encode_demand_distribution

# Simple Gaussian demand
mu, sigma = 3, 1
demand_pdf = lambda x: norm.pdf(x, mu, sigma)

# Create discrete version for verification
D_max = 5
demand_dist = {}
for d in range(D_max + 1):
    demand_dist[d] = norm.pdf(d, mu, sigma)
total = sum(demand_dist.values())
demand_dist = {d: p / total for d, p in demand_dist.items()}

print("Target demand distribution:")
for d in sorted(demand_dist.keys()):
    print(f"  d={d}: {demand_dist[d]:.4f}")

# Encode with FSL
# Use larger M for better approximation
M = 4  # More Fourier modes for better accuracy
print(f"\nEncoding with FSL (M={M})...")
ks, cs, meta = encode_demand_distribution(demand_pdf, D_max, M=M)

# Analyze encoding fidelity
print("\nAnalyzing FSL encoding fidelity...")
fsl_metrics = analyze_fsl_encoding(ks, cs, demand_dist, D_max, n_samples=10000)

print(f"\nResults:")
print(f"  Fidelity: {fsl_metrics['fidelity']:.6f} (target: 1.0)")
print(f"  TVD: {fsl_metrics['tvd']:.6f} (target: 0.0)")

print("\nEncoded distribution:")
for d in sorted(fsl_metrics['encoded_probs'].keys()):
    target = demand_dist.get(d, 0)
    encoded = fsl_metrics['encoded_probs'][d]
    error = encoded - target
    print(f"  d={d}: target={target:.4f}, encoded={encoded:.4f}, error={error:+.4f}")
