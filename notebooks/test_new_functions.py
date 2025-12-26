"""
新しく追加した可視化関数のテスト
"""
import numpy as np
from scipy.stats import norm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from newsvendor import (
    plot_cost_landscape,
    draw_qaoa_circuit,
    encode_demand_distribution
)

def test_cost_landscape():
    """コスト関数の曲線を描画するテスト"""
    print("=" * 70)
    print("Test 1: Cost Landscape Visualization")
    print("=" * 70)

    # ガウス分布の需要
    mu, sigma = 10, 3
    D_max = 31
    Q_max = 31
    c = 1.0
    lam = 20.0

    demand_pdf = lambda x: norm.pdf(x, mu, sigma)

    # コスト関数を可視化
    result = plot_cost_landscape(
        demand_dist=demand_pdf,
        c=c,
        lam=lam,
        Q_max=Q_max,
        D_max=D_max
    )

    print(f"\nReturned result keys: {list(result.keys())}")
    print(f"Optimal q: {result['optimal_q']}")
    print(f"Optimal cost: {result['optimal_cost']:.4f}")


def test_circuit_visualization():
    """量子回路の可視化をテスト"""
    print("\n" + "=" * 70)
    print("Test 2: QAOA Circuit Visualization")
    print("=" * 70)

    # 小さい問題で回路を描画
    mu, sigma = 3, 1
    D_max = 7
    Q_max = 7
    c = 1.0
    lam = 5.0

    demand_pdf = lambda x: norm.pdf(x, mu, sigma)

    # FSLエンコーディング
    M = 2
    ks, cs, meta = encode_demand_distribution(demand_pdf, D_max, M=M)

    # QAOAパラメータ
    gammas = [0.5, 0.3]  # p=2
    betas = [0.2, 0.4]

    n_q = int(np.ceil(np.log2(Q_max + 1)))
    n_d = int(np.ceil(np.log2(D_max + 1)))

    print(f"\nProblem size: Q_max={Q_max}, D_max={D_max}")
    print(f"Qubits: n_q={n_q}, n_d={n_d}")
    print(f"QAOA layers: p={len(gammas)}")

    # 回路を描画
    circuit = draw_qaoa_circuit(
        gammas=gammas,
        betas=betas,
        n_q=n_q,
        n_d=n_d,
        ks=ks,
        cs=cs,
        c=c,
        lam=lam,
        Q_max=Q_max,
        D_max=D_max,
        show_gates=True,
        max_gates=50
    )

    print(f"\nCircuit returned with {circuit.get_qubit_count()} qubits")
    print(f"and {circuit.get_gate_count()} gates")


if __name__ == "__main__":
    # Test 1: Cost landscape
    test_cost_landscape()

    # Test 2: Circuit visualization
    test_circuit_visualization()

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - cost_landscape.png")
    print("  - qaoa_circuit_diagram.png (if qulacsvis is installed)")
