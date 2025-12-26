"""
Minimal QAOA test - just check if the circuit builds without errors.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from newsvendor import solve_newsvendor_qaoa

# Very small problem
demand_dist = {0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2}

print("Testing QAOA with minimal problem (Q_max=3, D_max=3)...")
print(f"Demand distribution: {demand_dist}")

try:
    result = solve_newsvendor_qaoa(
        demand_dist=demand_dist,
        c=1.0,
        lam=5.0,
        Q_max=3,
        D_max=3,
        p=1,  # Single QAOA layer
        M=1,  # Very small Fourier truncation to fit in 2 qubits
        n_shots=100,  # Few shots for speed
        verbose=True
    )

    print("\n✓ Success!")
    print(f"Quantum solution: q = {result['quantum_solution']}")
    print(f"Classical solution: q = {result['classical_solution']}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
