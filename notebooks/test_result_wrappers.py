"""
resultから直接可視化できる新しいラッパー関数のテスト
"""
import numpy as np
from scipy.stats import norm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from newsvendor import (
    solve_newsvendor_qaoa,
    visualize_cost_from_result,
    visualize_circuit_from_result
)

def test_wrappers():
    """resultを使った新しいラッパー関数をテスト"""
    print("=" * 70)
    print("Test: Result-based Visualization Wrappers")
    print("=" * 70)

    # 小さい問題でQAOAを実行
    mu, sigma = 3, 1
    D_max = 7
    Q_max = 7
    demand_pdf = lambda x: norm.pdf(x, mu, sigma)

    print(f"\nRunning QAOA...")
    print(f"  Problem: Q_max={Q_max}, D_max={D_max}")
    print(f"  Demand: Gaussian(μ={mu}, σ={sigma})")

    result = solve_newsvendor_qaoa(
        demand_dist=demand_pdf,
        c=1.0,
        lam=5.0,
        Q_max=Q_max,
        D_max=D_max,
        p=2,
        M=2,
        n_shots=500,
        verbose=False
    )

    print(f"\nQAOA completed!")
    print(f"  Quantum solution: q = {result['quantum_solution']}")
    print(f"  Classical solution: q = {result['classical_solution']}")
    print(f"  Approximation ratio: {result['quantum_cost'] / result['classical_cost']:.4f}")

    # テスト1: コスト関数の可視化（resultから直接）
    print("\n" + "=" * 70)
    print("Test 1: Visualizing cost landscape from result")
    print("=" * 70)

    cost_result = visualize_cost_from_result(result)
    print(f"\n✓ Cost landscape visualized")
    print(f"  Optimal q: {cost_result['optimal_q']}")
    print(f"  Optimal cost: {cost_result['optimal_cost']:.4f}")

    # テスト2: 量子回路の可視化（resultから直接）
    print("\n" + "=" * 70)
    print("Test 2: Visualizing QAOA circuit from result")
    print("=" * 70)

    circuit = visualize_circuit_from_result(result, show_gates=True)
    print(f"\n✓ Circuit visualized")
    print(f"  Qubits: {circuit.get_qubit_count()}")
    print(f"  Gates: {circuit.get_gate_count()}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - cost_landscape.png")
    print("  - qaoa_circuit_diagram.png")
    print("\nUsage in Jupyter:")
    print("""
# QAOAを実行
result = solve_newsvendor_qaoa(...)

# resultから直接可視化
visualize_cost_from_result(result)
visualize_circuit_from_result(result)
""")

if __name__ == "__main__":
    test_wrappers()
